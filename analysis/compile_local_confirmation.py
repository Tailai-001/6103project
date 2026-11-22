"""Archive and independently evaluate the frozen local three-arm experiment.

Refuses incomplete/debug/source-mismatched runs. Checkpoints stay local; only
small logs, saved configurations, provenance and validation predictions are
copied into the evidence folder. Does not load official test data.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from evaluate import load_split, verify_checkpoint_log
from models.resnet import build_resnet18
from scripts.evaluate_member_c_val import evaluate

ARMS = ('e1', 'e4', 'e4r')
DISPLAY = dict(e1='E1', e4='E4', e4r='E4-R')
FROZEN_REF = '634b960'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_correctness(candidate, reference, labels):
    """Descriptive disagreements on the same selected validation images, no p-value."""
    a = np.asarray(candidate) == np.asarray(labels)
    b = np.asarray(reference) == np.asarray(labels)
    if a.shape != b.shape or not a.size:
        raise ValueError('Expected paired nonempty predictions')
    return dict(samples=int(a.size), both_correct=int((a & b).sum()),
                candidate_only_correct=int((a & ~b).sum()),
                reference_only_correct=int((~a & b).sum()), both_wrong=int((~a & ~b).sum()),
                cpu_top1_difference_pp=float(100*(a.astype(float)-b).mean()))


class RecordingModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.predictions = []

    def forward(self, inputs):
        logits = self.model(inputs)
        self.predictions.extend(logits.argmax(1).cpu().tolist())
        return logits


def main():
    folder = ROOT / 'results/local_audit'
    records = json.loads((folder / 'formal_runs.json').read_text())
    names = [f'local_confirm_{arm}_s43' for arm in ARMS]
    if [r['run_id'] for r in records] != names or any(r['status'] != 'completed' for r in records):
        raise ValueError('All three frozen formal runs must finish before compiling')
    torch.set_num_threads(4)
    data = load_split(ROOT / 'data/cifar100', ROOT / 'data/splits', 'validation')
    labels = np.asarray(data.dataset.targets)[data.indices]
    predictions, summaries = {}, []
    shared = None
    raw = folder / 'raw_logs'
    raw.mkdir(exist_ok=True)
    for arm, record in zip(ARMS, records):
        name = record['run_id']
        checkpoint_dir = ROOT / 'checkpoints/local_audit' / name
        log = ROOT / f'logs/local_audit/{name}.csv'
        rows = list(csv.DictReader(log.open(newline='')))
        if [int(r['epoch']) for r in rows] != list(range(1, 201)):
            raise ValueError(f'{name}: expected 200 continuous epochs')
        for row in rows:
            if int(row['train_samples']) != 45000:
                raise ValueError(f'{name}: incomplete or debug epoch')
            for key in ('train_loss', 'val_loss', 'val_accuracy', 'train_seconds', 'epoch_seconds'):
                if not math.isfinite(float(row[key])) or float(row[key]) < 0:
                    raise ValueError(f'{name}: invalid {key}')
            if float(row['val_accuracy']) > 1:
                raise ValueError(f'{name}: invalid accuracy')
        config = json.loads((checkpoint_dir / 'config.json').read_text())
        if config != yaml.safe_load((ROOT / record['config']).read_text()):
            raise ValueError(f'{name}: runtime config differs from frozen YAML')
        frozen = subprocess.check_output(['git', 'show', f'{FROZEN_REF}:{record["config"]}'], cwd=ROOT)
        if config != yaml.safe_load(frozen):
            raise ValueError(f'{name}: runtime config differs from pre-training commit')
        env = json.loads((checkpoint_dir / 'environment.json').read_text())
        if (env['debug_max_batches'] is not None or env['device'] != 'mps'
                or env['augmentation_device'] != 'cpu'):
            raise ValueError(f'{name}: unexpected execution route')
        signature = {key: config[key] for key in ('data', 'model', 'training')}
        signature.update(seed=config['experiment']['seed'],
                         torch=env['torch'], python=env['python'], split=env['split_sha256'])
        if shared is not None and signature != shared:
            raise ValueError('Training settings, environment or splits differ between arms')
        shared = signature
        recorded = {**env['source_sha256'], 'train.py': env['train_sha256']}
        if recorded != record['source_sha256']:
            raise ValueError(f'{name}: run source metadata mismatch')
        for path, sha in recorded.items():
            if digest(ROOT / path) != sha:
                raise ValueError(f'{name}: source changed: {path}')
        for path, sha in env['split_sha256'].items():
            if digest(ROOT / 'data/splits' / path) != sha:
                raise ValueError(f'{name}: split changed')
        state = torch.load(checkpoint_dir / 'best.pth', map_location='cpu', weights_only=True)
        verify_checkpoint_log(state, log, require_formal=True)
        if state['config'] != config:
            raise ValueError(f'{name}: checkpoint config mismatch')
        model = build_resnet18()
        model.load_state_dict(state['model_state_dict'])
        recording = RecordingModel(model).eval()
        loader = DataLoader(data, batch_size=128, shuffle=False, num_workers=0)
        total, accuracy, f1, loss = evaluate(recording, loader, torch.device('cpu'))
        if total != 5000 or len(recording.predictions) != total:
            raise ValueError('Incomplete validation evaluation')
        predictions[arm] = recording.predictions
        best = max(rows, key=lambda r: float(r['val_accuracy']))
        summaries.append(dict(
            arm=arm, run_id=name, seed=43, epochs=len(rows), best_epoch=int(best['epoch']),
            logged_best_top1_percent=100*float(best['val_accuracy']),
            logged_last_top1_percent=100*float(rows[-1]['val_accuracy']),
            logged_last10_mean_percent=100*np.mean([float(r['val_accuracy']) for r in rows[-10:]]),
            cpu_checkpoint_top1_percent=100*accuracy, cpu_checkpoint_macro_f1=f1,
            cpu_checkpoint_loss=loss, checkpoint_sha256=digest(checkpoint_dir / 'best.pth'),
            log_sha256=digest(log),
            cpu_minus_logged_correct_images=int(round(total*(accuracy-float(best['val_accuracy'])))),
            elapsed_seconds=record['elapsed_seconds'],
        ))
        shutil.copyfile(log, raw / log.name)
        meta = folder / 'run_metadata' / name
        meta.mkdir(parents=True, exist_ok=True)
        for filename in ('config.json', 'environment.json'):
            shutil.copyfile(checkpoint_dir / filename, meta / filename)
        print(f'{name}: logged best {summaries[-1]["logged_best_top1_percent"]:.2f}%; '
              f'CPU {100*accuracy:.2f}%; F1 {f1:.5f}', flush=True)
    contrasts = []
    by_arm = {row['arm']: row for row in summaries}
    for candidate, reference in [('e4', 'e1'), ('e4', 'e4r')]:
        a, b = by_arm[candidate], by_arm[reference]
        contrast = dict(candidate=candidate, reference=reference,
                        best_difference_pp=a['logged_best_top1_percent']-b['logged_best_top1_percent'],
                        last10_difference_pp=a['logged_last10_mean_percent']-b['logged_last10_mean_percent'])
        contrast.update(paired_correctness(predictions[candidate], predictions[reference], labels))
        contrasts.append(contrast)
    with (folder / 'validation_predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['validation_position', 'cifar_train_index', 'true_label', *ARMS])
        writer.writerows((i, int(index), int(label), *(predictions[arm][i] for arm in ARMS))
                         for i, (index, label) in enumerate(zip(data.indices, labels)))
    result = dict(scope='same-host new-seed validation confirmation; official test not evaluated',
                  independent_local_training_seeds=1, runs=summaries, contrasts=contrasts,
                  validation_indices_sha256=digest(ROOT / 'data/splits/val_indices.npy'),
                  evaluation_source_sha256={p: digest(ROOT / p) for p in
                      ['analysis/compile_local_confirmation.py', 'scripts/evaluate_member_c_val.py', 'evaluate.py']},
                  limitations=['Selected on validation; no official-test performance claim.',
                               'Validation prediction disagreements do not estimate training-seed uncertainty.',
                               'E4/E4-R final shapes are not exactly matched.',
                               'Old seed-42 runs use other hosts; do not pool as a controlled two-seed study.'])
    (folder / 'local_confirmation.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with (folder / 'local_confirmation.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(summaries)
    lines = ['# Local same-host confirmation: seed 43', '',
             '| Arm | Best epoch | Best val Top-1 | Last | Last-10 mean | CPU checkpoint Top-1 | CPU Macro-F1 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        lines.append(f'| {row["arm"].upper()} | {row["best_epoch"]} | '
                     f'{row["logged_best_top1_percent"]:.2f}% | {row["logged_last_top1_percent"]:.2f}% | '
                     f'{row["logged_last10_mean_percent"]:.3f}% | {row["cpu_checkpoint_top1_percent"]:.2f}% | '
                     f'{row["cpu_checkpoint_macro_f1"]:.5f} |')
    lines += ['', 'All arms: 200 epochs, fixed 45k/5k split, CPU augmentation + Apple MPS model.',
              'Earliest best validation checkpoint; independent CPU re-evaluation shown separately.',
              'No official test evaluation; one local training seed; final shapes remain imperfectly matched.', '',
              '| Candidate − reference | Best difference (pp) | Last-10 difference (pp) | CPU: candidate-only correct | CPU: reference-only correct |',
              '|---|---:|---:|---:|---:|']
    for row in contrasts:
        lines.append(f'| {row["candidate"].upper()} − {row["reference"].upper()} | '
                     f'{row["best_difference_pp"]:+.2f} | {row["last10_difference_pp"]:+.3f} | '
                     f'{row["candidate_only_correct"]} | {row["reference_only_correct"]} |')
    lines += ['', 'Prediction disagreements describe the same selected validation images, not seed-level statistical significance.']
    (folder / 'local_confirmation.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    # Raw per-epoch curves are descriptive; epochs are not independent repeats.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for arm, color in zip(ARMS, ['#777777', '#19698a', '#ca6b35']):
        rows = list(csv.DictReader((raw / f'local_confirm_{arm}_s43.csv').open()))
        x = [int(row['epoch']) for row in rows]
        y = [100*float(row['val_accuracy']) for row in rows]
        for ax in axes:
            ax.plot(x, y, label=DISPLAY[arm], color=color, lw=1.3)
    axes[0].set_title('Complete validation history')
    axes[1].set_title('Final 30 epochs')
    axes[1].set_xlim(171, 200)
    final = [100*float(row['val_accuracy']) for arm in ARMS
             for row in list(csv.DictReader((raw / f'local_confirm_{arm}_s43.csv').open()))[-30:]]
    axes[1].set_ylim(min(final)-.3, max(final)+.3)
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Top-1 (%)')
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Local same-host confirmation: frozen seed 43, 200 epochs per arm')
    for suffix in ('png', 'pdf'):
        fig.savefig(folder / f'local_validation_curves.{suffix}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
