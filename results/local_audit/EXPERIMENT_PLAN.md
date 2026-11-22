# Frozen local confirmation experiment

Written before starting any local formal training or inspecting its validation results.

- Purpose: add a same-host E1/E4/E4-R classification contrast using a new seed, rather than select another favourable validation configuration.
- Base code/evidence: `8408c039b0ad33b97ffc8f9214135f024ef71efb` (Member D, includes B/C).
- Host: Apple M5 Max, 64 GiB RAM. Python 3.11, torch 2.5.1, torchvision 0.20.1.
- Model device: Apple MPS float32; augmentation device: CPU, preserving the original float64 integral-image computations. All three arms use this same route and the same training loop.
- Dataset: the existing fixed 45,000/5,000 training/validation indices. Official test is not evaluated.
- Arms: E1 `random`; E4 `seam_dp`; E4-R `seam_random`.
- Seed: **43**, fixed before running. Batch 128, 200 epochs, SGD lr=0.1/momentum=0.9/weight_decay=0.0005, cosine scheduler, alpha=1, p=0.5, no mixed precision.
- E4/E4-R: K=4, gamma=0.5, beta=1, radius=2, seam_mu=0.05, min_width=2, area_budget=0.2; random_mu=0.225 from the existing training-only shape calibration. No local parameter search; do not substitute the post-hoc r=3 variant.
- Checkpoint: earliest best clean validation Top-1. Preserve all 200 epochs, last epoch and last-10 mean. Do not treat epochs as independent seed replicates.
- Sequential execution; no concurrent model training. Best/last checkpoints stay local and are not committed.
- Interpretation: this adds one new seed and one same-host three-arm comparison. It does not establish multi-seed statistical significance or official-test performance. The existing E4-R shape mismatch remains a limitation.
- If a run fails, record its status and reason; do not replace a run based on its accuracy.

Pre-formal execution amendment: use `num_workers=0` for every local arm. The three initial real-data smoke runs used 4 workers and exposed repeated macOS worker startup overhead (~50 seconds per 2-batch smoke). No formal classification run had started when this loader-only choice was made. Preserve those original smoke records. CPU augmentation RNG stream equivalence across policies is separately regression-tested.

Pre-formal backend correction: strict multi-step CPU/MPS ResNet comparison exposed an asynchronous host-transfer race in the initial torch 2.5 MPS preflight. Use blocking host transfers on MPS; the existing CUDA path is unchanged. The original 3 smoke outputs are excluded from numerical evidence, and a new synchronized CPU-augmentation/MPS smoke passed (train loss 4.7050; 256 training samples). None is a formal accuracy result. Current suite: 131 passed, 1 CUDA-only skip.
