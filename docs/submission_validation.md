# Submission validation

Acceptance checks performed on **22 November 2026**, using a newly created isolated Python environment on an Apple M5 Max. Environment: Python 3.11.16, torch 2.5.1, torchvision 0.20.1, NumPy 1.26.4 and pytest 9.2.0, installed from the delivered requirement files. The official dataset was reused from an existing local archive; no new dataset download or long training was performed for these checks.

| Check | Observed result |
|---|---|
| Dependency consistency | All installed packages compatible |
| Standard-library repository verifier | 10 upstream file hashes, 2 fixed split hashes, Python syntax and current documentation links pass |
| Complete pytest suite | **133 passed, 1 skipped**; the skipped check requires unavailable CUDA hardware |
| Synthetic CPU integration | All **7 policies** pass ResNet-18 optimizer steps, final-mask weights, clean evaluation and checkpoint save/reload |
| Synthetic MPS integration | All **7 policies** pass with CPU augmentation and an MPS model |
| Actual CIFAR-100 CPU integration | All **7 policies** pass using batches from the fixed training/validation loaders |
| Dataset/split preparation | Official archive integrity, 45k/5k disjoint partition, canonical hashes and 450/50 examples per class pass |
| Root training CLI | E4 short configuration runs 2 batches, validates all 5,000 images and writes best/last checkpoints, settings, environment and CSV |
| Root checkpoint evaluation CLI | Reloads the short-run best checkpoint, verifies its selected epoch/log and evaluates the fixed 5,000-image validation split |
| Evidence reproduction | Original 18 runs plus 3 confirmation runs and four numeric figures regenerate without data or weights |
| CSV consistency | Rebuilt main, ablation, efficiency and boundary CSVs match all four committed tables byte-for-byte |
| Confirmation metrics | All three best-validation Top-1 and Macro-F1 values agree with the public 5,000-image predictions |
| Historical source audit | All 18 original formal-run evidence checks pass, including recorded B/C historical source hashes |
| Archive preservation | The original A handoff and official CutMix snapshot are unchanged relative to the pre-integration `main` |

Smoke checks establish execution, not classifier performance. Their partial-training metrics are not added to the formal experiment tables. The 21 formal results remain the previously recorded 200-epoch experiments.

[GitHub Actions](https://github.com/Tailai-001/6103project/actions/workflows/ci.yml) repeats the dataset-free checks on a clean Linux CPU runner: installation, repository validation, the complete tests, seven-policy smoke execution, result reproduction and the historical source audit. The run history records its actual status. Local checks above do not imply that Windows or CUDA execution was rerun.

The current [README](../README.md), [running guide](running.md) and [implemented protocol](experiment_protocol.md) are the delivery instructions. Dataset files, model weights and course documents containing student details are deliberately outside the public code package; every command needing them explains how to obtain or generate its inputs.
