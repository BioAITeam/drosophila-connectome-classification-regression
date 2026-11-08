"""Prepare the same grouped Iris partitions and train-fitted scaling for all models."""
from common import HERE, REPO, SPLIT_SEEDS, save_json, load_json, sha256, relative, partition_path
import importlib.metadata
import os
import numpy as np
from flybrain.datasets import prepare_dataset


def main():
    if importlib.metadata.version('scikit-learn')!='1.7.2':
        raise RuntimeError('Use scikit-learn 1.7.2, matching the original experiments')
    records=[]
    for seed in SPLIT_SEEDS:
        coverage=[]
        for fold in range(5):
            data=prepare_dataset('iris',seed=seed,outer_fold=fold)
            assert data['raw_x'].shape==(150,4)
            assert len(np.unique(data['groups']))==149
            assert np.bincount(data['raw_y']).tolist()==[50,50,50]
            sets=[set(data['groups'][data['indices'][p]]) for p in ['train','val','test']]
            assert all(not(a & b) for i,a in enumerate(sets) for b in sets[i+1:])
            assert sorted(np.concatenate(list(data['indices'].values())).tolist())==list(range(150))
            arrays={key:data[key] for key in ['raw_x','raw_y','groups']}
            for part in ['train','val','test']:
                arrays['index_'+part]=data['indices'][part]
                arrays['x_'+part]=data['x_'+part]
                arrays['y_'+part]=data['y_'+part]
                assert len(np.unique(data['y_'+part]))==3
            path=partition_path(seed,fold)
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                with np.load(path,allow_pickle=False) as previous:
                    assert set(previous.files)==set(arrays)
                    assert all(np.array_equal(previous[k],v) for k,v in arrays.items()),'Existing partition differs'
            else:
                temp=path.with_suffix('.npz.tmp')
                with temp.open('wb') as f:np.savez_compressed(f,**arrays)
                temp.replace(path)
            metadata={**data['metadata'],'npz_sha256':sha256(path),'path':relative(path),
                      'source_module':'3 Connectome/flybrain/datasets.py',
                      'source_sha256':sha256(REPO/'3 Connectome/flybrain/datasets.py')}
            save_json(path.with_suffix('.json'),metadata)
            records.append({'split_seed':seed,'fold':fold,'path':relative(path),'sha256':metadata['npz_sha256'],'sizes':metadata['sizes']})
            coverage.extend(arrays['index_test'].tolist())
        assert sorted(coverage)==list(range(150))
    save_json(HERE/'data/index.json',{'dataset':'iris','samples':150,'features':4,'groups':149,'partitions':records})
    print('Prepared and verified 10 grouped partitions for 150 Iris flowers.',flush=True)

if __name__=='__main__':main()
