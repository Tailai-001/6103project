"""Register and run B jobs through A's single training entry point."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import threading
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import string


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['smoke','pilot','formal'],default='formal')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--suffix',default='')
    args=parser.parse_args()
    if any(c not in string.ascii_letters+string.digits+'-_' for c in args.suffix):
        parser.error('suffix may contain only letters, digits, hyphens and underscores')
    if args.workers<1 or (args.phase!='formal' and args.workers!=1):
        parser.error('workers must be positive; smoke/pilot run in isolation')
    root=Path(__file__).resolve().parents[1]
    registry=root/'experiments/registry.csv'
    archive=root/'source_initial.tar.gz'
    source_hash=hashlib.sha256(archive.read_bytes()).hexdigest() if archive.exists() else 'see per-run source_sha256'
    split_hash=';'.join(f'{p.name}:{hashlib.sha256(p.read_bytes()).hexdigest()}' for p in sorted((root/'data/splits').glob('*.npy')))
    with registry.open() as f: fields=next(csv.reader(f))
    if args.phase=='formal':
        jobs=[(f'b_{policy}_k{k}_s42',policy,k,[]) for policy,k in
              [('dual_edge',1),('dual_edge',4),('source_edge',4),('dual_edge',8),('source_edge',8)]]
    else:
        extra=['--epochs','2','--max-batches','2'] if args.phase=='smoke' else ['--epochs','5']
        jobs=[('b_'+args.phase,'dual_edge',4,extra)]
    query=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True)
    device_name=query.stdout.strip().split('\n')[0] if query.returncode==0 else 'CPU'
    lock=threading.Lock()

    def run_job(job):
        name,policy,k,extra=job
        if args.suffix: name += "_"+args.suffix
        config_path=f'configs/b_{policy}_k{k}_s42.yaml'
        with lock:
            with registry.open() as f: existing=list(csv.DictReader(f))
            if (any(x['run_id']==name for x in existing) or
                    (root/f'logs/{name}.csv').exists() or (root/f'logs/{name}.stdout.log').exists() or
                    (root/f'checkpoints/{name}').exists()):
                raise FileExistsError(f'Existing run {name}; preserve it and choose a fresh run name')
        row=dict(run_id=name,experiment='E2' if policy=='dual_edge' else 'E2-S',policy=policy,seed=42,
            git_commit='',config_path=config_path,split_sha256=split_hash,
            environment=f'checkpoints/{name}/environment.json',
            device=device_name,
            started_at=datetime.now(timezone.utc).isoformat(),status='running',log_uri=f'logs/{name}.csv',
            checkpoint_uri=f'checkpoints/{name}/best.pth',
            notes=f'phase={args.phase}; source_archive_sha256={source_hash}; overrides={extra}; workers={args.workers}; validation only; walltime may include GPU contention')
        with lock:
            with registry.open('a',newline='') as f: csv.DictWriter(f,fieldnames=fields).writerow(row)
        start=time.perf_counter()
        with (root/f'logs/{name}.stdout.log').open('x') as log:
            result=subprocess.run([sys.executable,'-u','-B','train.py','--config',config_path,'--run-name',name,*extra],
                                  cwd=root,stdout=log,stderr=subprocess.STDOUT)
        row['status']='completed' if result.returncode==0 else 'failed'
        row['total_training_seconds']=time.perf_counter()-start
        row['notes']+=f'; exit_code={result.returncode}'
        diagnostic_path=root/'results/member_b/diagnostics.json'
        if diagnostic_path.exists():
            diagnostic=json.loads(diagnostic_path.read_text())
            for record in diagnostic['results']:
                if record['policy']==policy and record['k']==k:
                    row['augmentation_ms_per_batch']=record['augmentation_ms_per_batch']
                    row['notes']+='; augmentation timing isolated before training'
        log_path=root/f'logs/{name}.csv'
        if log_path.exists():
            with log_path.open() as f: metrics=list(csv.DictReader(f))
            if metrics: row['val_top1_percent']=100*max(float(x['val_accuracy']) for x in metrics)
        with lock:
            with registry.open() as f: rows=list(csv.DictReader(f))
            rows=[row if x['run_id']==name else x for x in rows]
            temp=registry.with_suffix('.tmp')
            with temp.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            temp.replace(registry)
        print(json.dumps(row),flush=True)
        return result.returncode

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes=list(pool.map(run_job,jobs))
    if any(codes): raise SystemExit(next(code for code in codes if code))


if __name__=='__main__': main()
