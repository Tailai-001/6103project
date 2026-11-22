# Local same-host confirmation: seed 43

| Arm | Best epoch | Best val Top-1 | Last | Last-10 mean | CPU checkpoint Top-1 | CPU Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E1 | 192 | 79.92% | 79.78% | 79.824% | 79.92% | 0.79818 |
| E4 | 195 | 79.78% | 79.62% | 79.546% | 79.78% | 0.79768 |
| E4R | 197 | 80.08% | 79.84% | 79.846% | 80.08% | 0.79964 |

All arms: 200 epochs, fixed 45k/5k split, CPU augmentation + Apple MPS model.
Earliest best validation checkpoint; independent CPU re-evaluation shown separately.
No official test evaluation; one local training seed; final shapes remain imperfectly matched.

| Candidate − reference | Best difference (pp) | Last-10 difference (pp) | CPU: candidate-only correct | CPU: reference-only correct |
|---|---:|---:|---:|---:|
| E4 − E1 | -0.14 | -0.278 | 275 | 282 |
| E4 − E4R | -0.30 | -0.300 | 257 | 272 |

Prediction disagreements describe the same selected validation images, not seed-level statistical significance.
