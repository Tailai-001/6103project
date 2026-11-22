# Running tools

Use these portable commands from the repository root after installing `requirements-dev.txt`:

| Command | Purpose |
|---|---|
| `python scripts/verify_repository.py` | Standard-library checks of the upstream snapshot, split hashes, syntax and current documentation links |
| `python scripts/smoke_test.py` | All seven policies, shared ResNet-18 training steps, evaluation and temporary checkpoint roundtrip; no dataset |
| `python scripts/prepare_data.py --download` | Explicit official-data download and fixed split/class-balance verification |
| `python scripts/smoke_test.py --data-root /path/to/cifar100` | The same integration checks using existing real train/validation data |
| `python analysis/reproduce_results.py` | Rebuild public result tables and figures without data or weights |

The root `train.py` and `evaluate.py` are the delivery entry points. [Complete running guide](../docs/running.md).

Historical experiment tools are retained: `member_b_suite.py` and `member_c_suite.py` schedule recorded configurations; diagnostics generate paired images, Sobel/seam maps and synchronized timings; checkpoint evaluators reload locally available weights; summaries compile their logs. B/C shell and SLURM wrappers have server-specific path/environment assumptions, so a grader should use the portable root commands above. Their optional shutdown behavior is not part of the quickstart.

`run_local_confirmation.py` records the frozen seed-43 three-arm experiment with CPU augmentation and an MPS model through the shared trainer. Its [original plan](../results/local_audit/EXPERIMENT_PLAN.md) and [completed evidence](../RESULTS_AUDIT.md) are preserved; smoke runs are never formal results.
