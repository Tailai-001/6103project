#!/usr/bin/env bash
# One-shot runner for C's formal training on a CUDA machine.
# Run from anywhere inside the repo checkout:   bash scripts/run_member_c.sh
#
# Env vars:
#   PY               python interpreter (default: .venv/bin/python, else python)
#   C_ARMS           arms to train (default: "e3 e4 e4r"; see ARMS in member_c_suite.py)
#   C_RUN_SUFFIX     suffix for run names, required to re-run without overwriting
#   C_SKIP_SMOKE     set to 1 to skip the smoke run  (it also runs only once per suffix)
#   C_SKIP_PILOT     set to 1 to skip the 5-epoch pilot (it also runs only once per suffix)
#   C_AUTO_SHUTDOWN  set to 1 to pack results when the script exits and shut the machine
#                    down ONLY if everything succeeded (a failure keeps the machine)
#   C_MACHINE        label used in the results archive name (default: hostname)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python )}
ARMS=${C_ARMS:-"e3 e4 e4r"}
SUFFIX=${C_RUN_SUFFIX:-}
MACHINE=${C_MACHINE:-$(hostname)}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
mkdir -p logs results/member_c

finish() {
  status=$?
  set +e
  echo "== exit status $status"
  if [ "${C_AUTO_SHUTDOWN:-0}" = 1 ]; then
    archive="../member_c_results_${MACHINE}.tar.gz"
    # Logs, configs/environments, registry and results; .pth weights stay on the data disk.
    tar --exclude='*.pth' -czf "$archive" logs checkpoints experiments/registry.csv results/member_c \
        logs_run_member_c.txt 2>/dev/null || true
    echo "== packed $(cd .. && pwd)/member_c_results_${MACHINE}.tar.gz"
    if [ "$status" != 0 ]; then
      # Never release the machine on failure: on AutoDL the GPU may be taken by someone else.
      echo "== FAILED: NOT shutting down. Read logs/*.stdout.log, fix, then shut down by hand."
      return
    fi
    echo "== all done; shutting down in 60 s (Ctrl+C to cancel)"
    # Ctrl+C during the wait cancels the shutdown instead of skipping ahead to it.
    if sleep 60; then shutdown; else echo "== shutdown cancelled"; fi
  fi
}
trap finish EXIT

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
if [ "${C_SKIP_SMOKE:-0}" != 1 ] && [ ! -e "logs/c_smoke$tag.csv" ]; then
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

echo "== checkpoint re-evaluation + Macro-F1 (validation only, every C run with a best.pth here)"
"$PY" -B scripts/evaluate_member_c_val.py --suffix "$SUFFIX" --num-workers 4 \
  | tee results/member_c/checkpoint_validation_stdout.txt

echo "== summary"
"$PY" -B scripts/summarize_member_c.py --suffix "$SUFFIX" | tee results/member_c/summary_stdout.txt
