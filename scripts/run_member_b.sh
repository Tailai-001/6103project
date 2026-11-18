#!/usr/bin/env bash
# Run from a dedicated project directory inside the user's authorized remote root.
set -euo pipefail
cd "$(dirname "$0")/.."
allowed_root=${B_ALLOWED_ROOT:?Set B_ALLOWED_ROOT to your authorized workspace before running}
allowed_root=$(cd "$allowed_root" && pwd -P)
case "$(pwd -P)/" in "$allowed_root/"*) ;; *) echo 'Refusing writes outside authorized workspace' >&2; exit 1;; esac
export TMPDIR="$PWD/tmp" XDG_CACHE_HOME="$PWD/cache" PIP_CACHE_DIR="$PWD/cache/pip"
export MPLCONFIGDIR="$PWD/cache/matplotlib" CUDA_CACHE_PATH="$PWD/cache/cuda"
export TORCH_HOME="$PWD/cache/torch" TRITON_CACHE_DIR="$PWD/cache/triton" TORCHINDUCTOR_CACHE_DIR="$PWD/cache/inductor" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" logs results/member_b
PY="$PWD/.venv/bin/python"
"$PY" -B -c 'import torch; assert torch.cuda.is_available(), "GPU suite requires working CUDA"; print(torch.__version__, torch.cuda.get_device_name(0))'
"$PY" -B -m pip freeze > results/member_b/requirements.txt
"$PY" -B -m unittest discover -s tests -v > results/member_b/tests.txt 2>&1
"$PY" -B -c 'from torchvision.datasets import CIFAR100; CIFAR100("data/cifar100",train=True,download=True)'
"$PY" -B scripts/member_b_diagnostics.py > results/member_b/diagnostics_stdout.txt 2>&1
"$PY" -u -B scripts/member_b_suite.py --phase smoke --suffix "${B_RUN_SUFFIX:-}"
"$PY" -u -B scripts/member_b_suite.py --phase pilot --suffix "${B_RUN_SUFFIX:-}"
# Registry runner records each job before it starts and failures on nonzero exit.
"$PY" -B -c 'import csv,sys; name="b_pilot"+("_"+sys.argv[1] if sys.argv[1] else ""); r=list(csv.DictReader(open("logs/"+name+".csv"))); job=next(x for x in csv.DictReader(open("experiments/registry.csv")) if x["run_id"]==name); estimate=200*float(job["total_training_seconds"])/len(r); print("Estimated formal run seconds:",estimate); assert len(r)==5 and estimate<19800, "Pilot requires review before formal training"' "${B_RUN_SUFFIX:-}"
"$PY" -u -B scripts/member_b_suite.py --workers "${B_RUN_WORKERS:-1}" --suffix "${B_RUN_SUFFIX:-}"
"$PY" -B scripts/summarize_member_b.py --suffix "${B_RUN_SUFFIX:-}" > results/member_b/summary_stdout.txt

"$PY" -B scripts/evaluate_member_b_val.py --suffix "${B_RUN_SUFFIX:-}" > results/member_b/checkpoint_validation_stdout.txt
