"""Prepare and run the Iris topology and traditional-classifier campaign."""
from __future__ import annotations
from common import HERE, REPO, SOURCE, GRAPH_SEEDS, SPLIT_SEEDS, OPTIMIZATION_SEEDS, save_json, load_json, relative, sha256, now
import argparse
from collections import Counter, deque
import fcntl
import importlib.metadata
import itertools
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def make_plan():
    from ml_models import MODEL_NAMES, build_spec
    graphs=[('anatomical','anatomical',None)]
    for seed in GRAPH_SEEDS:
        graphs.extend([(f'rewired-s{seed}','rewired',seed),(f'random-s{seed}','random',seed)])
    g_jobs=[];m_jobs=[]
    for graph_id,family,seed in graphs:
        for split,opt,fold in itertools.product(SPLIT_SEEDS,OPTIMIZATION_SEEDS,range(5)):
            g_jobs.append(dict(id=f'graph__{graph_id}__split-{split}__seed-{opt}__fold-{fold}',kind='graph',
                               graph_id=graph_id,family=family,graph_seed=seed,split_seed=split,optimization_seed=opt,fold=fold))
    for split,opt,fold in itertools.product(SPLIT_SEEDS,OPTIMIZATION_SEEDS,range(5)):
        for model in MODEL_NAMES:
            m_jobs.append(dict(id=f'ml__{model}__split-{split}__seed-{opt}__fold-{fold}',kind='ml',model=model,
                               split_seed=split,optimization_seed=opt,fold=fold))
    grids={name:build_spec(name,'iris',4,3,0).settings for name in MODEL_NAMES}
    configuration={'schema_version':1,'dataset':'Iris classification','training':'fresh fits, no reused checkpoints',
      'split_seeds':list(SPLIT_SEEDS),'optimization_seeds':list(OPTIMIZATION_SEEDS),'outer_folds':5,
      'inner_folds':4,'validation_inner_fold':0,'graph_seeds':list(GRAPH_SEEDS),
      'graph_training':{'epochs':120,'batch_size':16,'learning_rate':0.03,'gradient_clip':1.0,
         'optimizer':'Adam','beta1':0.9,'beta2':0.999,'epsilon':1e-8,'min_delta':1e-8,
         'steps':24,'dt':0.5,'leak':1.0,'input_amplitude':2.0,'temperature':0.02,
         'gain_bounds':[0.05,1.95],'epoch_zero_eligible':True,'selection':'maximum validation balanced accuracy, earliest tie'},
      'ml_grids':grids,'ml_selection':'GridSearchCV with one PredefinedSplit, validation balanced accuracy, first exact tie, no train+validation refit',
      'counts':{'anatomical_graphs':1,'rewired_graphs':30,'random_graphs':30,'graph_fits':len(g_jobs),
                'ml_selected_fits':len(m_jobs),'ml_search_fits':20*sum(s['candidate_count'] for s in grids.values()),
                'final_fits':len(g_jobs)+len(m_jobs)},
      'analysis':{'bootstrap_samples':10000,'bootstrap_seed':13007,'effect':'anatomical minus control in percentage points',
                  'conditioning':'fixed trained models and four OOF repetitions, graph and duplicate-group resampling'}}
    jobs=[];gs=deque(g_jobs);ms=deque(m_jobs)
    while gs or ms:
        for _ in range(2):
            if gs:jobs.append(gs.popleft())
        if ms:jobs.append(ms.popleft())
    for filename,value in [('configuration.json',configuration),('jobs.json',{'schema_version':1,'jobs':jobs})]:
        path=HERE/filename
        if path.exists() and load_json(path)!=value:raise ValueError(f'Existing immutable plan differs: {filename}')
        if not path.exists():save_json(path,value)
    assert len(jobs)==1500 and len({j['id'] for j in jobs})==1500
    return jobs


def provenance():
    paths=list(SOURCE.glob('*.py'))+[HERE/'jobs.json',HERE/'configuration.json']
    # Freeze the manifests that bind each graph and partition to its content.
    # Individual workers verify the referenced NPZ hashes before fitting.
    paths += [HERE/'graphs/index.json',HERE/'data/index.json']
    paths += list((HERE/'data').glob('split-*-fold-*.json'))
    paths += [REPO/'3 Connectome/flybrain'/name for name in ['model.py','controls.py','datasets.py','dataset_specs.py','metrics.py']]
    return {'files':{relative(p):sha256(p) for p in sorted(paths)},
            'packages':{name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn','torch','xgboost','joblib']},
            'python':sys.version.split()[0],'cpu_threads_per_job':1,'device':'cpu'}


def complete(job):
    folder=HERE/'runs'/job['id']
    if not(folder/'complete.json').exists():return False
    record=load_json(folder/'complete.json')
    if record.get('id',record.get('job',{}).get('id'))!=job['id']:raise ValueError('Completion job identity mismatch')
    for name in ['result.json','predictions.npz','best.pt' if job['kind']=='graph' else 'best.joblib']:
        if not(folder/name).is_file():raise ValueError(f'Incomplete result: {job["id"]}/{name}')
    if record.get('software_validation_only') or record.get('smoke'):
        raise ValueError(f'Smoke output cannot complete a production job: {job["id"]}')
    if record.get('result_sha256')!=sha256(folder/'result.json'):
        raise ValueError(f'Result checksum differs: {job["id"]}')
    result=load_json(folder/'result.json')
    if any(result.get(key)!=value for key,value in job.items()):
        raise ValueError(f'Result job identity differs: {job["id"]}')
    if result.get('software_validation_only') or result.get('smoke'):
        raise ValueError(f'Smoke result cannot complete a production job: {job["id"]}')
    if result.get('predictions_sha256')!=sha256(folder/'predictions.npz'):
        raise ValueError(f'Predictions checksum differs: {job["id"]}')
    checkpoint=folder/('best.pt' if job['kind']=='graph' else 'best.joblib')
    if record.get('checkpoint_sha256')!=sha256(checkpoint) or result.get('checkpoint_sha256')!=record['checkpoint_sha256']:
        raise ValueError(f'Checkpoint checksum differs: {job["id"]}')
    return True


def summary(jobs,finished,failed,running,started,state):
    categories={}
    for job in jobs:
        group=job['family'] if job['kind']=='graph' else job['model']
        entry=categories.setdefault(group,{'total':0,'completed':0,'running':0,'failed':0,'pending':0})
        entry['total']+=1
        key='completed' if job['id'] in finished else ('failed' if job['id'] in failed else ('running' if job['id'] in running else 'pending'))
        entry[key]+=1
    return {'state':state,'updated_utc':now(),'started_utc':started,'controller_pid':os.getpid(),
            'total':len(jobs),'completed':len(finished),'failed':len(failed),'running':len(running),
            'pending':len(jobs)-len(finished)-len(failed)-len(running),'groups':categories,
            'active_jobs':[{'id':key,'pid':value[0].pid,'elapsed_seconds':time.monotonic()-value[2]} for key,value in running.items()]}


def stop_children(running):
    """Reap every child before releasing the controller lock, even on errors."""
    for process,_,_,_ in running.values():
        if process.poll() is None:
            try:process.terminate()
            except ProcessLookupError:pass
    deadline=time.monotonic()+15
    for process,log,_,_ in running.values():
        try:
            try:process.wait(timeout=max(0.0,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        finally:log.close()
    running.clear()


def run(workers,retry_failed):
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE/'campaign.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This campaign already has an active controller')
        return run_locked(workers,retry_failed)


def run_locked(workers,retry_failed):
    jobs=load_json(HERE/'jobs.json')['jobs']
    source=provenance();path=HERE/'provenance.json'
    if path.exists() and load_json(path)!=source:raise RuntimeError('Campaign source or software changed since launch')
    if not path.exists():save_json(path,source)
    expected=set(j['graph_id'] for j in jobs if j['kind']=='graph')
    graph_index=load_json(HERE/'graphs/index.json')['graphs']
    if expected!={g['id'] for g in graph_index}:raise RuntimeError('Graph inventory differs from registered jobs')
    finished={j['id'] for j in jobs if complete(j)}
    failed={} if retry_failed or not(HERE/'failures.json').exists() else load_json(HERE/'failures.json')
    job_ids={job['id'] for job in jobs}
    failed={k:v for k,v in failed.items() if k in job_ids and k not in finished}
    save_json(HERE/'failures.json',failed)
    pending=deque(j for j in jobs if j['id'] not in finished and j['id'] not in failed)
    running={};started=now();stopping=False;last_update=0
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    (HERE/'logs').mkdir(exist_ok=True)
    save_json(HERE/'controller.json',{'pid':os.getpid(),'started_utc':started,'workers':workers,'source':relative(HERE)})
    analysis_running={};analysis_code=None
    try:
        with (HERE/'events.jsonl').open('a',buffering=1) as events:
            def event(kind,**fields):
                import json
                events.write(json.dumps({'utc':now(),'event':kind,**fields})+'\n')
            event('controller_started',workers=workers,resumed_completed=len(finished))
            while (pending or running) and not stopping:
                while pending and len(running)<workers and not stopping:
                    job=pending.popleft();job_id=job['id']
                    log=(HERE/'logs'/f'{job_id}.log').open('a',buffering=1)
                    script=SOURCE/('run_graph_experiments.py' if job['kind']=='graph' else 'run_ml_experiments.py')
                    try:
                        process=subprocess.Popen([sys.executable,'-u','-B',str(script),'--job-id',job_id],cwd=REPO,
                                                 stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
                    except BaseException:
                        log.close()
                        raise
                    running[job_id]=(process,log,time.monotonic(),job)
                    event('started',id=job_id,pid=process.pid,job_kind=job['kind'])
                for job_id,(process,log,t0,job) in list(running.items()):
                    code=process.poll()
                    if code is None:continue
                    log.close();del running[job_id]
                    if code==0 and complete(job):
                        finished.add(job_id);event('completed',id=job_id,elapsed_seconds=time.monotonic()-t0)
                    else:
                        failed[job_id]={'returncode':code,'utc':now(),'log':relative(HERE/'logs'/f'{job_id}.log')}
                        save_json(HERE/'failures.json',failed);event('failed',id=job_id,returncode=code)
                if time.monotonic()-last_update>=10:
                    state=summary(jobs,finished,failed,running,started,'running')
                    save_json(HERE/'status.json',state)
                    print(f'{now()} completed={len(finished)}/{len(jobs)} running={len(running)} failed={len(failed)} pending={len(pending)}',flush=True)
                    last_update=time.monotonic()
                time.sleep(1)
            if stopping:stop_children(running)
            if not stopping:
                save_json(HERE/'status.json',summary(jobs,finished,failed,running,started,'analyzing'))
                command=[sys.executable,'-u','-B',str(SOURCE/'analyze_results.py'),'--bootstrap','10000','--seed','13007']
                if failed:command.append('--partial')
                with (HERE/'logs/analysis.log').open('w') as log:
                    process=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
                    analysis_running['analysis']=(process,log,time.monotonic(),None)
                    event('analysis_started',pid=process.pid)
                    while process.poll() is None and not stopping:time.sleep(0.5)
                    if stopping:stop_children(analysis_running)
                    analysis_code=process.wait()
                    analysis_running.clear()
                event('analysis_finished',returncode=analysis_code)
            state=('interrupted' if stopping else 'analysis_failed' if analysis_code else
                   'completed_with_errors' if failed else 'completed')
            current=summary(jobs,finished,failed,running,started,state)
            if analysis_code is not None:
                current.update(analysis_returncode=analysis_code,analysis_log=relative(HERE/'logs/analysis.log'))
            save_json(HERE/'status.json',current)
            event('controller_finished',state=state,completed=len(finished),failed=len(failed))
        return 1 if failed or (analysis_code and not stopping) else 0
    except BaseException as exc:
        stop_children(running)
        stop_children(analysis_running)
        try:
            state=summary(jobs,finished,failed,running,started,'controller_failed')
            state['controller_error']=f'{type(exc).__name__}: {str(exc).replace(str(REPO), "<repo>")}'
            save_json(HERE/'status.json',state)
        except Exception:pass
        raise
    finally:
        stop_children(running)
        stop_children(analysis_running)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true',help='Create immutable plans only, without training')
    parser.add_argument('--workers',type=int,default=24)
    parser.add_argument('--retry-failed',action='store_true')
    args=parser.parse_args()
    if args.workers<1 or args.workers>48:parser.error('workers must be between 1 and 48')
    if args.prepare:
        make_plan();print(load_json(HERE/'configuration.json')['counts']);return
    raise SystemExit(run(args.workers,args.retry_failed))

if __name__=='__main__':main()
