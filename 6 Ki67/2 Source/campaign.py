"""Independent GPU workers with atomic claims on a shared job queue."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from .common import ROOT, OUTPUT, CLASS_TASKS, digest, destination, now, portable_metadata, read_json, write_json, resolve_output_reference, writable_location

def _validated_gpu_indices(indices, setting):
    if (not isinstance(indices, list) or not indices
            or any(type(gpu) is not int or gpu < 0 for gpu in indices)
            or len(indices) != len(set(indices))):
        raise ValueError(f"{setting} requires a nonempty list of distinct nonnegative physical GPU indices.")
    return set(indices)


def read_execution_policy(config, output):
    """Read a mutable scheduling policy without changing frozen experiment settings."""
    if output is None:
        return None
    path = Path(output) / "status/execution_policy.json"
    if not path.exists():
        return None
    if "joint" in config:
        raise ValueError("Execution-policy overrides are not supported for joint campaigns.")
    policy = read_json(path)
    if (not isinstance(policy, dict) or type(policy.get("schema_version")) is not int
            or policy["schema_version"] != 1):
        raise ValueError("Execution policy requires schema_version 1.")
    _validated_gpu_indices(policy.get("allowed_gpus"), "Execution policy allowed_gpus")
    excluded = policy.get("excluded_training_job_ids")
    if (not isinstance(excluded, list) or any(not isinstance(job, str) for job in excluded)
            or len(excluded) != len(set(excluded))):
        raise ValueError("Execution policy requires distinct excluded training job IDs.")
    training_ids = {job["id"] for job in expand(config) if job["kind"] == "train"}
    if not set(excluded).issubset(training_ids):
        raise ValueError("Execution policy excludes jobs outside the frozen training campaign.")
    return policy


def excluded_training_jobs(config, output):
    policy = read_execution_policy(config, output)
    return set(policy["excluded_training_job_ids"]) if policy else set()


def allowed_gpus(config, output=None):
    """Validate GPU assignments for separate-dataset and joint campaigns."""
    policy = read_execution_policy(config, output)
    if policy is not None:
        return set(policy["allowed_gpus"])
    if "joint" in config and (not isinstance(config["joint"], dict) or set(config["datasets"]) != {"joint"}):
        raise ValueError("Joint plans must contain only the joint dataset.")
    runtime = config.get("runtime")
    indices = runtime.get("gpu_indices") if isinstance(runtime, dict) else None
    return _validated_gpu_indices(indices, "runtime.gpu_indices")

def validate_context(config, output):
    """Validate a joint destination before creating artifacts or reserving a GPU."""
    resolved = writable_location(output)
    allowed_gpus(config, resolved)
    if "joint" in config:
        expected = config.get("output_root")
        if expected is None or resolved != resolve_output_reference(expected) or resolved == OUTPUT.resolve():
            raise ValueError("Joint plans require --output matching their dedicated output_root; the original output root is forbidden.")
        source = resolve_output_reference(config["joint"]["source_output"])
        if source == resolved or source.is_relative_to(resolved):
            raise ValueError("The joint output must not contain or replace its source output.")
    return resolved

def expand(config):
    jobs = []
    representations = config["representations"]
    if "joint" in config:
        representations = sorted(representations, key=lambda rep: rep != "phikonv2")
    for dataset in config["datasets"]:
        for rep in representations:
            if rep != "rgb":
                jobs.append(dict(kind="extract", dataset=dataset, representation=rep))
    for dataset in config["datasets"]:
        for rep in representations:
            for graph in config["graphs"]:
                jobs.append(dict(kind="profile", dataset=dataset, representation=rep, graph=graph))
    if "joint" in config:
        tasks = sorted(config["tasks"], key=lambda task: (task != "multiclass", task not in CLASS_TASKS))
        task_seeds = [(task, seed) for task in tasks for seed in config["seeds"]]
    else:
        task_seeds = [(task, seed) for seed in config["seeds"] for task in config["tasks"]]
    for task, seed in task_seeds:
        for dataset in config["datasets"]:
            for rep in representations:
                for graph in config["graphs"]:
                    jobs.append(dict(kind="train", dataset=dataset, representation=rep,
                                     graph=graph, task=task, seed=seed))
    for job in jobs:
        job["id"] = "-".join(str(job[k]) for k in ["kind", "dataset", "representation", "graph", "task", "seed"] if k in job)
    return jobs

def _profile_file(output, job):
    return Path(output)/"status/profiles"/f"{job['dataset']}-{job['representation']}-{job['graph']}.json"

def done(output, job):
    if job["kind"] == "extract":
        return (Path(output)/"features"/job["dataset"]/job["representation"]/"metadata.json").exists()
    if job["kind"] == "profile":
        return _profile_file(output, job).exists()
    from .training import run_path
    return (run_path(output, job)/"complete.json").exists()

def ready(output, job):
    if job["kind"] == "extract":
        return True
    if job["representation"] != "rgb" and not (Path(output)/"features"/job["dataset"]/job["representation"]/"metadata.json").exists():
        return False
    return job["kind"] == "profile" or _profile_file(output, job).exists()

@contextmanager
def claim(path):
    handle = destination(path).open("a+")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            acquired = False
        yield acquired
    finally:
        handle.close()

def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, PermissionError, IndexError):
        return None

class Heartbeat:
    def __init__(self, output, gpu):
        self.path = Path(output)/"status"/f"worker-gpu{gpu}.json"
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.state = {"pid": os.getpid(), "process_start_identity": process_identity(os.getpid()),
                      "gpu": gpu, "gpu_record": json.loads(os.environ.get("FLYBRAIN_GPU_RECORD", "null")),
                      "started_at": now(), "state": "starting", "last_progress": now()}
        self.thread = threading.Thread(target=self.loop, daemon=True)
    def __enter__(self):
        self.thread.start()
        return self
    def update(self, **fields):
        with self.lock:
            self.state.update(fields)
            self.state["last_progress"] = now()
    def publish(self):
        with self.lock:
            snapshot = {**self.state, "heartbeat": now()}
        write_json(self.path, snapshot)
    def loop(self):
        while not self.stop.is_set():
            self.publish()
            self.stop.wait(15)
    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.publish()

def worker(config, output):
    output = validate_context(config, output)
    gpu = int(os.environ["KI67_PHYSICAL_GPU"])
    if gpu not in allowed_gpus(config, output):
        raise RuntimeError("Worker GPU is not authorized by this campaign.")
    import torch
    from . import features
    from .training import train_job, profile
    device = os.environ["FLYBRAIN_DEVICE"]
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Campaign workers require the explicitly reserved CUDA GPU.")
    torch.set_num_threads(config["runtime"]["threads_per_worker"])
    jobs = expand(config)
    jobs.sort(key=lambda job: {"extract": 0, "profile": 1, "train": 2}[job["kind"]])
    with Heartbeat(output, gpu) as heartbeat:
        while True:
            excluded = excluded_training_jobs(config, output)
            pending = [job for job in jobs if job["id"] not in excluded and not done(output, job)]
            failed = [job for job in pending if (Path(output)/"status/errors"/f"{job['id']}.json").exists()]
            if not pending:
                heartbeat.update(state="complete", job=None)
                return
            ran = False
            for job in pending:
                if job in failed or not ready(output, job):
                    continue
                with claim(Path(output)/"status/claims"/f"{job['id']}.lock") as acquired:
                    if (not acquired or job["id"] in excluded_training_jobs(config, output)
                            or done(output, job)):
                        continue
                    heartbeat.update(state="running", job=job, phase="starting", epoch=None, batch=None)
                    heartbeat.publish()
                    try:
                        if job["kind"] == "extract":
                            result = features.extract(config, output, job["dataset"], job["representation"],
                                                      device=device, progress=lambda info: heartbeat.update(**info))
                        elif job["kind"] == "profile":
                            result = profile(config, output, job["dataset"], job["representation"],
                                             job["graph"], device, heartbeat.update)
                        else:
                            result = train_job(config, output, job, device, heartbeat.update)
                        write_json(Path(output)/"status/completed_jobs"/f"{job['id']}.json",
                                   {"job": job, "finished_at": now(), "gpu": gpu, "result": result})
                        heartbeat.update(state="idle", job=None)
                    except Exception as exc:
                        record = {"job": job, "timestamp": now(), "gpu": gpu, "error": str(exc), "traceback": traceback.format_exc()}
                        write_json(Path(output)/"status/errors"/f"{job['id']}.json", record)
                        print(record["traceback"], file=sys.stderr, flush=True)
                        heartbeat.update(state="job_error", error=str(exc))
                    torch.cuda.empty_cache()
                    ran = True
                    break
            if not ran:
                # Do not hold a GPU indefinitely when every remaining dependency failed.
                possible = [j for j in pending if j not in failed and ready(output, j)]
                if not possible and failed:
                    heartbeat.update(state="blocked_by_failed_jobs", pending=len(pending), failures=len(failed))
                    return
                heartbeat.update(state="waiting_for_jobs", pending=len(pending))
                time.sleep(15)

def status(config, output):
    all_jobs = expand(config)
    policy = read_execution_policy(config, output)
    excluded = set(policy["excluded_training_job_ids"]) if policy else set()
    jobs = [job for job in all_jobs if job["id"] not in excluded]
    def count(items):
        values = {kind: {"total": 0, "complete": 0, "failed": 0} for kind in ["extract", "profile", "train"]}
        for job in items:
            complete = done(output, job)
            group = values[job["kind"]]
            group["total"] += 1
            group["complete"] += int(complete)
            group["failed"] += int((Path(output)/"status/errors"/f"{job['id']}.json").exists() and not complete)
        return values
    counts = count(jobs)
    workers = []
    for path in sorted((Path(output)/"status").glob("worker-gpu*.json")):
        try:
            value = read_json(path)
            actual = process_identity(value["pid"])
            value["alive"] = actual is not None and actual == value.get("process_start_identity", actual)
            workers.append(value)
        except (ValueError, KeyError):
            continue
    estimates = []
    for job in jobs:
        if job["kind"] != "train" or done(output, job):
            continue
        path = _profile_file(output, job)
        if not path.exists():
            continue
        seconds = read_json(path)["estimated_epoch_seconds"]
        epochs = config["training"]["classification_epochs"] if job["task"] in CLASS_TASKS else config["training"]["regression_epochs"]
        estimates.append(seconds * (epochs+1))
    result = {"timestamp": now(), "counts": counts, "workers": workers,
              "profile_based_remaining_gpu_hours_ceiling": sum(estimates)/3600 if estimates else None,
              "estimated_jobs_covered": len(estimates),
              "estimate_note": "Initial small-batch profiles; regression early stopping and completed epochs may reduce work. Not a measured completion guarantee."}
    if policy is not None:
        result.update(execution_policy=policy, original_counts=count(all_jobs),
                      excluded_counts=count([job for job in all_jobs if job["id"] in excluded]),
                      counts_scope="Eligible jobs after execution-policy exclusions; exclusions are not completions.")
    return result

def launch(config, output, gpus, background=True):
    output = validate_context(config, output)
    with claim(Path(output)/"status/launch.lock") as acquired:
        if not acquired:
            raise RuntimeError("Another launch is in progress; consult status before retrying.")
        return _launch_locked(config, output, gpus, background)

def restart_workers(output):
    """Stop only this launch's identity-verified Ki67 workers; preserve all artifacts."""
    import signal
    path = Path(output)/"status/launch.json"
    if not path.exists():
        return []
    stopped = []
    for worker in read_json(path).get("workers", []):
        pid, identity = worker["pid"], worker.get("process_start_identity")
        if identity is None or process_identity(pid) != identity:
            continue
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if b"ki67.bootstrap" not in command or b"worker" not in command:
            raise RuntimeError(f"Refusing to signal PID {pid}: not a Ki67 worker.")
        os.kill(pid, signal.SIGTERM)
        stopped.append(worker)
    deadline = time.monotonic()+30
    while any(process_identity(w["pid"]) == w["process_start_identity"] for w in stopped):
        if time.monotonic() >= deadline:
            raise RuntimeError("A worker has not exited; no additional process was started.")
        time.sleep(.25)
    return [w["pid"] for w in stopped]

def _launch_locked(config, output, gpus, background):
    config = portable_metadata(config)
    authorized = allowed_gpus(config, output)
    if not gpus or not set(gpus).issubset(authorized) or len(gpus) != len(set(gpus)):
        raise ValueError(f"Use distinct physical GPUs from {','.join(map(str, sorted(authorized)))}.")
    frozen = Path(output)/"manifests/campaign.json"
    if frozen.exists() and read_json(frozen) != config:
        raise ValueError("Existing frozen campaign differs; use a separate output subfolder.")
    write_json(frozen, config)
    write_json(Path(output)/"manifests/jobs.json", expand(config))
    previous_path = Path(output)/"status/launch.json"
    previous = read_json(previous_path) if previous_path.exists() else {}
    records = []
    for gpu in gpus:
        previous_worker = next((w for w in previous.get("workers", []) if w["gpu"] == gpu), None)
        if previous_worker and previous_worker.get("process_start_identity") is not None and process_identity(previous_worker["pid"]) == previous_worker["process_start_identity"]:
            records.append({**previous_worker, "existing": True})
            continue
        state_path = Path(output)/"status"/f"worker-gpu{gpu}.json"
        if state_path.exists():
            state = read_json(state_path)
            if process_identity(state.get("pid", 0)) == state.get("process_start_identity") and state.get("process_start_identity") is not None:
                records.append({"gpu": gpu, "pid": state["pid"], "process_start_identity": state["process_start_identity"], "existing": True})
                continue
        log = destination(Path(output)/"logs"/f"worker-gpu{gpu}.log")
        with log.open("ab", buffering=0) as handle:
            proc = subprocess.Popen([str(ROOT/"run.sh"), "--device", "cuda", "--gpu", str(gpu),
                                     "worker", "--output", str(output), "--plan", str(frozen)],
                                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                    start_new_session=background)
        records.append({"gpu": gpu, "pid": proc.pid, "process_start_identity": process_identity(proc.pid), "log": str(log)})
        write_json(previous_path, {**previous, "workers": records, "launch_in_progress": True})
    monitor_log = destination(Path(output)/"logs/monitor.log")
    monitor_pid = previous.get("monitor_pid")
    monitor_start = previous.get("monitor_process_start_identity")
    if monitor_start is None or process_identity(monitor_pid or 0) != monitor_start:
        with monitor_log.open("ab", buffering=0) as handle:
            monitor = subprocess.Popen([str(ROOT/"run.sh"), "monitor", "--output", str(output), "--plan", str(frozen)],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=background)
        monitor_pid, monitor_start = monitor.pid, process_identity(monitor.pid)
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    first_status = datetime.now(ZoneInfo("America/Santiago")) + timedelta(hours=1)
    result = {"started_at": now(), "workers": records, "monitor_pid": monitor_pid,
              "monitor_process_start_identity": monitor_start,
              "suggested_first_status_local": first_status.isoformat(), "timezone": "America/Santiago"}
    write_json(Path(output)/"status/launch.json", result)
    return result

def monitor(config, output):
    last_snapshot = 0.0
    while True:
        result = status(config, output)
        write_json(Path(output)/"status/latest.json", result)
        if time.monotonic()-last_snapshot >= config["runtime"]["status_interval_seconds"]:
            stamp = now().replace(":", "-")
            write_json(Path(output)/"status/snapshots"/f"{stamp}.json", result)
            from .evaluation import summarize
            summarize(output)
            last_snapshot = time.monotonic()
        completed = result["counts"]["train"]["complete"] == result["counts"]["train"]["total"]
        alive = any(x["alive"] for x in result["workers"])
        if completed or (not alive and time.monotonic()-last_snapshot > 90):
            from .evaluation import summarize
            summarize(output)
            write_json(Path(output)/"status/final.json", {**result, "complete": completed})
            return
        time.sleep(30)
