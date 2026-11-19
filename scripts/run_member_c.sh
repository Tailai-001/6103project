#!/usr/bin/env bash
# One-shot runner for C's formal training (E3 / E4 / E4-R) on a CUDA machine.
# Run from anywhere inside the repo checkout:   bash scripts/run_member_c.sh
#
# Env vars:
#   PY            python interpreter (default: .venv/bin/python, else python)
#   C_ARMS        arms to train (default: "e3 e4 e4r")
#   C_RUN_SUFFIX  suffix for run names, required to re-run without overwriting
#   C_SKIP_PILOT  set to 1 to skip the 5-epoch pilot (it also runs only once per suffix)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python )}
ARMS=${C_ARMS:-"e3 e4 e4r"}
SUFFIX=${C_RUN_SUFFIX:-}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
mkdir -p logs results/member_c

echo "== environment"
"$PY" -B -c 'import torch; assert torch.cuda.is_available(), "Formal training requires CUDA"; print(torch.__version__, torch.cuda.get_device_name(0))'
"$PY" -B -m pip freeze > results/member_c/requirements.txt 2>/dev/null || true
git rev-parse HEAD > results/member_c/git_commit.txt

echo "== tests"
if "$PY" -c 'import pytest' 2>/dev/null; then
  "$PY" -B -m pytest -q tests > results/member_c/tests.txt 2>&1 || { tail -20 results/member_c/tests.txt; exit 1; }
  tail -1 results/member_c/tests.txt
else
  echo "pytest not installed; skipping tests (pip install pytest to enable)"
fi

echo "== data (downloads the CIFAR-100 train set into data/cifar100 if missing; git-ignored)"
"$PY" -B -c 'from torchvision.datasets import CIFAR100; CIFAR100("data/cifar100", train=True, download=True)'

echo "== augmentation timing (GPU, synchronized)"
"$PY" -B scripts/member_c_aug_timing.py | tee results/member_c/aug_timing_stdout.txt

tag=${SUFFIX:+_$SUFFIX}
# Smoke and pilot run once per suffix; later jobs (e.g. one SLURM job per arm) skip them.
if [ ! -e "logs/c_smoke$tag.csv" ]; then
  echo "== smoke"
  "$PY" -u -B scripts/member_c_suite.py --phase smoke --suffix "$SUFFIX"
fi

if [ "${C_SKIP_PILOT:-0}" != 1 ] && [ ! -e "logs/c_pilot$tag.csv" ]; then
  echo "== pilot (5 epochs of E4)"
  "$PY" -u -B scripts/member_c_suite.py --phase pilot --suffix "$SUFFIX" | tee results/member_c/pilot_stdout.txt
fi

echo "== formal: $ARMS (sequential, 200 epochs each)"
# shellcheck disable=SC2086
"$PY" -u -B scripts/member_c_suite.py --phase formal --arms $ARMS --suffix "$SUFFIX"

echo "== summary"
"$PY" -B scripts/summarize_member_c.py --suffix "$SUFFIX" | tee results/member_c/summary_stdout.txt
