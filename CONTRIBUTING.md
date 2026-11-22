# Contribution and experiment records

Read the [README](README.md), [implemented protocol](docs/experiment_protocol.md) and relevant module README. The root code is the integrated seven-policy delivery; the original A directory and upstream snapshot are archives.

1. Work on a focused branch and submit a pull request. Codex-created branches use `codex/`.
2. Before a code change is merged, run `python scripts/verify_repository.py`, `python -m pytest -q` and `python scripts/smoke_test.py`. Reproduce tables when changing evidence processing.
3. Preserve the single training loop, fixed split, final-mask label weights and shared proposal/RNG rules. Document any intentional protocol change and keep it distinct from historical runs.
4. Record a unique run ID, complete settings, source version, seed, environment and raw logs for new experiments. Keep failures; never overwrite evidence or relabel smoke/pilot runs as formal training.
5. Keep `third_party/CutMix-PyTorch/` unchanged and retain attribution. Do not commit datasets, large checkpoints, credentials, personal identifiers or course documents containing student details.

The current evidence comprises 18 original seed-42 runs and 3 same-host seed-43 confirmation runs. All reported classification results are validation measurements. Additional official-test or multi-seed work must be recorded as new measurements.

Public read/clone access requires no collaborator invitation. Team members who need write access can be invited by the owner or contribute through a fork and pull request.
