"""Sequential local same-host E1/E4/E4-R confirmation; never evaluates official test."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('e1', 'e4', 'e4r')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    target = ROOT / 'results/local_audit'
    private = ROOT / 'outputs/local_confirmation'
    private.mkdir(parents=True, exist_ok=True)
    record = target / ('smoke_runs.json' if args.smoke else 'formal_runs.json')
    if record.exists():
        raise FileExistsError(f'Preserve existing execution record: {record}')
    files = [ROOT / 'train.py'] + [p for d in ('augmentations', 'models', 'data') for p in sorted((ROOT/d).glob('*.py'))]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    records = []
    env = dict(os.environ, OMP_NUM_THREADS='4', VECLIB_MAXIMUM_THREADS='4', PYTHONHASHSEED='0')
    for arm in ARMS:
        name = f'local_confirm_{arm}_s43' + ('_smoke' if args.smoke else '')
        config = f'configs/local_confirm_{arm}_s43.yaml'
        command = [sys.executable, '-u', 'train.py', '--config', config, '--run-name', name,
                   '--device', 'mps', '--augmentation-device', 'cpu']
        if args.smoke:
            command += ['--epochs', '1', '--max-batches', '2']
        for p in files:
            if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[p.relative_to(ROOT).as_posix()]:
                raise RuntimeError('Core source changed after experiment start')
        item = dict(run_id=name, config=config, command=command[2:], status='running',
                    started_unix=time.time(), source_sha256=hashes)
        records.append(item)
        record.write_text(json.dumps(records, indent=2)+'\n')
        print(f'START {name}', flush=True)
        with (private/(name+'.stdout.txt')).open('w') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        item.update(status='completed' if process.returncode==0 else 'failed', exit_code=process.returncode,
                    elapsed_seconds=time.time()-item['started_unix'])
        record.write_text(json.dumps(records, indent=2)+'\n')
        print(f'{item["status"].upper()} {name}: {item["elapsed_seconds"]:.1f}s', flush=True)
        if process.returncode:
            raise SystemExit(f'{name} failed; evidence retained in outputs/local_confirmation')
        with (ROOT/'logs/local_audit'/f'{name}.csv').open() as log:
            rows=list(csv.DictReader(log))
        print(f'  epochs={len(rows)} best_val={100*max(float(r["val_accuracy"]) for r in rows):.2f}%',flush=True)


if __name__=='__main__':
    main()
