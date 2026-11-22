# Running and reproducing the project

Use the integrated root-level entry points. The [README](../README.md) contains the shortest installation and execution sequence; [experiment_protocol.md](experiment_protocol.md) records the measured setup.

## Environment

The delivery is tested with 64-bit Python 3.11, torch 2.5.1, torchvision 0.20.1 and NumPy 1.26.4. Install the matching torch/torchvision pair from [PyTorch's official version instructions](https://pytorch.org/get-started/previous-versions/#v251), then install `requirements-dev.txt`. That file includes pytest; installing only `requirements.txt` supplies runtime dependencies.

On macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Choose one torch installation appropriate for the machine:

```bash
# macOS (CPU and supported Apple MPS)
python -m pip install torch==2.5.1 torchvision==0.20.1
# Linux/Windows CPU
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
# Linux/Windows NVIDIA GPU with a compatible driver
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

Then install and check the environment:

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
python -c "import torch, torchvision, numpy; print(torch.__version__, torchvision.__version__, numpy.__version__)"
```

For Windows PowerShell, create the environment using `py -3.11 -m venv .venv`. Activate with `.venv\Scripts\Activate.ps1`, or invoke `.venv\Scripts\python.exe` directly wherever the commands say `python`. The local acceptance record covers macOS; Linux CPU is checked by GitHub Actions. Windows commands are provided for setup and have not been locally exercised.

## First checks

```bash
python scripts/verify_repository.py
python scripts/smoke_test.py --out outputs/smoke.json
python -m pytest -q
python analysis/reproduce_results.py --out-dir outputs/reproduced
```

These commands need no dataset, pretrained weights or external service after dependencies are installed. The smoke test uses synthetic four-image batches, executes all seven policies with ResNet-18, and deletes its temporary checkpoint automatically. The result reproducer uses the small public evidence already in this repository, and refuses the committed `results/` tree as its output directory.

Use pytest for the **complete** suite. `unittest discover` alone does not collect C's pytest-style seam/DP tests. Hardware-dependent tests skip explicitly. To limit CPU thread overhead on a small machine, use `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q` on macOS/Linux; the smoke runner has its own `--threads` option.

## Prepare CIFAR-100

```bash
# Explicit download, official training data and fixed split checks
python scripts/prepare_data.py --download
# Alternatively, check an archive that is already present
python scripts/prepare_data.py --data-root /path/to/cifar100
```

The expected extracted directory is `data/cifar100/cifar-100-python/`, containing the official CIFAR-100 archive files. `torchvision` checks their integrity. Although the official download contains both partitions, preparation reads training labels only and performs no test evaluation.

The supplied [split files](../data/splits/README.md) are the reproducibility authority. `python -m data.split` validates them without rewriting them or requiring scikit-learn. The original generation script is retained in the archived A handoff; regenerating the split is unnecessary for every delivered workflow.

For an existing dataset, run `python scripts/smoke_test.py --data-root /path/to/cifar100` to check the actual train/validation loader with all seven methods. The command never downloads data. For a complete training run at a nondefault location, copy the chosen YAML and change `data.root`; do not change the fixed split.

## Training

The [short CPU configuration](../configs/smoke_cpu.yaml) has one epoch, batch 4 and zero loader workers. With `--max-batches 2`, it processes just two training batches; validation remains the full 5,000-image split. The normal configurations retain 200 epochs and batch 128.

```bash
python train.py --config configs/smoke_cpu.yaml --max-batches 2 --device cpu --run-name smoke_e4
python train.py --config configs/c_seam_dp_k4_s42.yaml --run-name e4_s42
```

`--device auto` is the default and selects CUDA if available, otherwise CPU. For an available Apple MPS device:

```bash
python train.py --config configs/c_seam_dp_k4_s42.yaml --device mps --augmentation-device cpu --run-name e4_mps_s42
```

The source algorithms accumulate integral scores in float64, so their augmentation stays on CPU for MPS. This preserves the original math and a single training loop. In the smoke command, `--device mps` selects that arrangement automatically. Checkpoint evaluation through `evaluate.py` supports CPU and CUDA; CPU can evaluate checkpoints trained on MPS.

Outputs for a run named `e4_s42` are:

```text
logs/e4_s42.csv
checkpoints/e4_s42/best.pth
checkpoints/e4_s42/last.pth
checkpoints/e4_s42/config.json
checkpoints/e4_s42/environment.json
```

An existing run name raises `FileExistsError`; choose a new name such as `e4_s42_repeat1`. Checkpoint selection uses clean validation accuracy, with the earliest maximum retained. There is no resume option. Full runs can take hours depending on the machine; the quick checks are sufficient to verify code execution without repeating the research training budget.

## Checkpoint evaluation

```bash
python evaluate.py --checkpoint checkpoints/e4_s42/best.pth --run-id e4_s42 --log logs/e4_s42.csv --split validation --device cpu --out outputs/e4_s42_validation.json
```

The weights must come from an actual local run; example paths are output paths, not files hosted in the repository. This evaluates the saved model and checks the checkpoint's selected epoch against its CSV. Metrics include Top-1 percentage, Macro-F1 and mean loss. `--data-root` and `--split-dir` are available for nondefault data locations.

Only after freezing all selection choices, a complete formal run may be evaluated with `--split test`. It requires the training CSV and verifies completion with 45,000 training samples per epoch. There are currently no reported official-test scores.

## Evidence reproduction

`analysis/reproduce_results.py` rebuilds the original 18-run table, three confirmation summaries, boundary tables, efficiency tables and four plots. Confirmation accuracy and Macro-F1 are recomputed from public saved predictions; it does not claim to reload unpublished checkpoints. Each output manifest identifies its input hashes.

Optional further audits:

```bash
# Requires a full Git clone, not a shallow clone or source ZIP
python analysis/audit_training_evidence.py --out outputs/training_evidence_audit.json
# Recompute the paired boundary bootstrap from committed CSV data
python analysis/paired_boundary_evidence.py --out outputs/paired_boundary --bootstrap 4000
```

The original B/C suites and SLURM/cloud-server templates remain available for provenance and batch scheduling. They have server-specific environment assumptions; the direct root `train.py` commands are the portable delivery interface. Optional cloud shutdown flags in historical runners are disabled by default and are not needed for any command in this guide.

## Troubleshooting

| Symptom | Resolution |
|---|---|
| `No matching distribution` for torch | Use the documented Python 3.11 environment and matching wheel platform; this pinned pair is not a Python 3.13+ setup. |
| `operator torchvision::nms does not exist` | Recreate the environment with matching torch/torchvision versions from the same CPU or CUDA channel. |
| `No module named pytest` | Install `requirements-dev.txt`, not only the runtime requirements. |
| Dataset not found or corrupted | Run `scripts/prepare_data.py --download` explicitly, or pass the directory containing `cifar-100-python`. |
| Loader subprocess errors | Use the smoke configuration with `data.num_workers: 0`, or copy a formal YAML and set workers to 0. |
| MPS float64 error | Add `--augmentation-device cpu` when calling `train.py --device mps`. |
| Run already exists | Choose a fresh `--run-name`; preserved logs/checkpoints are not overwritten. |
| Historical source audit cannot find a commit | Use `git fetch --unshallow` if the clone is shallow, or obtain a normal full clone. Other quick checks do not need history. |
