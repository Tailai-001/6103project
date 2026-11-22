"""Independently validate the numerical and source evidence behind all 18 runs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import yaml

from compile_results import A, B, C, ROOT, build_runs, read_csv, read_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/local_audit/training_evidence_audit.json')
    args=parser.parse_args()
    train=np.load(ROOT/'data/splits/train_indices.npy',allow_pickle=False)
    val=np.load(ROOT/'data/splits/val_indices.npy',allow_pickle=False)
    if (train.shape!=(45000,) or val.shape!=(5000,) or len(np.unique(train))!=45000
        or len(np.unique(val))!=5000 or np.intersect1d(train,val).size
        or not np.array_equal(np.sort(np.r_[train,val]),np.arange(50000))):
        raise ValueError('Train/validation are not the expected disjoint complete partition')
    registry={r['run_id']:r for r in read_csv(ROOT/'experiments/registry.csv')}
    reports=[]
    runs=build_runs()
    for row,logs in runs:
        for line in logs:
            for key in ('learning_rate','train_loss','val_loss','val_accuracy','train_seconds','epoch_seconds'):
                value=float(line[key])
                if not math.isfinite(value) or value<0 or (key=='val_accuracy' and value>1):
                    raise ValueError(f'{row["run_id"]}: invalid {key}')
        report=dict(run_id=row['run_id'],member=row['source'],epochs=len(logs),seed=row['seed'],
                    machine=row['machine'],best_val_top1_percent=row['best_val_top1_percent'],
                    last10_mean_percent=row['last10_val_top1_mean_percent'],
                    log_sha256=digest(ROOT/row['log_path']),checks=['continuous_200_epochs','45000_train_samples_per_epoch',
                    'finite_metrics','published_best_and_last10_consistency','shared_split_hashes'])
        if row['source'] in ('B','C'):
            folder=B if row['source']=='B' else C
            meta=folder/'run_metadata'/row['run_id']
            config=read_json(meta/'config.json')
            env=read_json(meta/'environment.json')
            registered=registry[row['run_id']]
            published_cfg=yaml.safe_load((ROOT/registered['config_path']).read_text())
            if config!=published_cfg:
                raise ValueError(f'{row["run_id"]}: saved config differs from published config')
            if env['debug_max_batches'] is not None:
                raise ValueError('Debug run was included as formal evidence')
            ref=registered.get('git_commit') or '9562fdffa1e42ea87c29890faeaf28fda07747dc'
            hashes={**env['source_sha256'],'train.py':env['train_sha256']}
            for path,expected in hashes.items():
                actual=hashlib.sha256(subprocess.check_output(['git','show',f'{ref}:{path}'],cwd=ROOT)).hexdigest()
                if expected!=actual:
                    raise ValueError(f'{row["run_id"]}: source hash mismatch for {path} at {ref}')
            report.update(training_ref=ref,source_files_verified=len(hashes),config_sha256=digest(meta/'config.json'))
            report['checks'] += ['saved_config_equals_published_config','not_debug','historical_source_hashes_verified']
        else:
            report['limitations']=['No saved per-run source hashes or per-run checkpoint validation in A handoff.']
        reports.append(report)
    by_id={r['run_id']:r for r in reports}
    pairs=[('c_e4_k4_s42','c_e3_k4_s42'),('c_e4_k4_s42','c_e4r_k4_s42'),
           ('c_e4_k4_s42','c_ctrl_e1_s42'),('c_e4_r3_k4_s42','c_e4_k4_s42'),
           ('c_e3_b2_k4_s42','c_ctrl_e2_k4_s42'),('b_source_edge_k4_s42_seq','b_dual_edge_k1_s42_seq')]
    contrasts=[]
    for candidate,reference in pairs:
        a,b=by_id[candidate],by_id[reference]
        contrasts.append(dict(candidate=candidate,reference=reference,same_host=a['machine']==b['machine'],
                              independent_training_seeds=1,
                              best_difference_pp=round(a['best_val_top1_percent']-b['best_val_top1_percent'],5),
                              last10_difference_pp=round(a['last10_mean_percent']-b['last10_mean_percent'],5),
                              significance='not estimated: one training seed; epochs are correlated'))
    result=dict(base_commit='8408c039b0ad33b97ffc8f9214135f024ef71efb',runs=reports,
                split=dict(train=45000,validation=5000,disjoint=True,complete_partition=True),
                total_formal_runs=len(reports),unique_seeds=sorted({r['seed'] for r in reports}),
                checkpoint_files_locally_available=False,source_hashes_independently_verified=True,
                contrasts=contrasts,
                limitations=['Log/config/source consistency does not prove that the historical training was executed.',
                             'Historical checkpoints are not in the repository; their reported re-evaluations were not independently repeated here.',
                             'No official-test result is inferred from validation logs.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'Verified {len(reports)} complete runs; split disjointness, logs/configs and B/C historical source hashes.')
    print(json.dumps(contrasts,indent=2))


if __name__=='__main__':
    main()
