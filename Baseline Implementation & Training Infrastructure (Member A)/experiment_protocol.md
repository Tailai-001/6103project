# AI6103 Seam-Aware CutMix — Experiment Protocol & A→D Handoff

**Prepared:** 2026-11-16  
**Owner:** Member A (Baseline & Training Infrastructure)  
**Status:** v0.1 — E0/E1 completed with seed 42; B/C integrations pending.  
**Authority:** Project README supplied by the team. The linked `docs/experiment_protocol.md` was not available for independent inspection when this handoff was prepared; compare this draft against that file before freezing the protocol. Values labeled *A decision* are decisions proposed/used by Member A, not requirements claimed to come from GitHub.

## 1. Non-negotiable rules from the project README

- Dataset CIFAR-100, official train split must be divided into train and validation; official test split is for final evaluation only.
- Four-stage CIFAR ResNet-18 (BasicBlock `[2,2,2,2]`, 3×3 stride-1 stem, no initial maxpool).
- E0=`none`, E1=`random`, E2=`dual_edge`, E2-S=`source_edge`, E3=`seam_aware`, E4=`seam_dp`, E4-R=`seam_random`.
- Same base training pipeline and fair budget for comparison. Only validation data for tuning / checkpoint selection.
- For mixed images use per-sample actual area: `lambda_actual = 1 - mask.sum()/(H*W)`; recompute from the *final* mask. Mix cross-entropy with `lambda_actual` and `1-lambda_actual`.
- Input to Sobel / RGB seam scoring remains in `[0,1]`, before model normalization. E1–E3 should share sampled original rectangles, candidate rule and image pairing; choose by score, not independent source/target crop locations.
- Record full configurations, seeds, device, code revision or source snapshot, checkpoints, failures, runtime, mask diagnostics and methods actually completed.
- E4 claims require the matched E4-R control and area/perimeter/fallback diagnostics.

## 2. A decisions used in existing E0/E1 formal runs (seed 42)

| Field | Value | Status |
|---|---|---|
| Data split | 45,000 train / 5,000 validation, stratified; 10,000 official test withheld | A decision implemented |
| Split seed | 42, saved `data/splits/train_indices.npy` and `val_indices.npy` | A decision implemented |
| Train transformations | RandomCrop(32, padding=4), RandomHorizontalFlip, ToTensor | Implemented |
| Val transformations | ToTensor | Implemented |
| Model normalization | CIFAR-100 mean `(0.5071,0.4867,0.4408)` and std `(0.2675,0.2565,0.2761)` after optional mixing | Implemented |
| Model | Four-stage CIFAR ResNet-18, randomly initialized, 100-way head | Required architecture, implemented |
| Epochs | 200 | A decision, completed E0/E1 |
| Batch size | 128 | A decision, completed E0/E1 |
| Optimizer | SGD, momentum 0.9, weight decay 0.0005 | A decision, completed E0/E1 |
| LR / schedule | Initial 0.1, cosine annealing over 200 epochs | A decision, completed E0/E1 |
| E1 CutMix | Beta(alpha=1, alpha=1); p=0.5 | A decision, completed |
| Primary metric | Clean validation Top-1 Accuracy; best checkpoint selected by maximum validation accuracy | README rule, implemented |
| Seed | 42 | First completed seed only; not multi-seed evidence |
| Hardware | NVIDIA A40 (46 GB), PyTorch 2.9.1+cu126, torchvision 0.24.1+cu126, Python 3.11.18 | Verified from submitted job output |
| GPU limit | 6 hours / job | Cluster limit reported by team |

### Verified seed-42 results from uploaded formal CSV files

| Run | Policy | Completed epochs | Best epoch | Best validation accuracy | Validation loss at best-accuracy epoch | Sum of epoch durations |
|---|---|---:|---:|---:|---:|---:|
| `e0_final_s42` | `none` | 200 | 186 | 77.74% | 0.9128 | 36.10 min |
| `e1_final_s42` | `random` | 200 | 200 | 80.52% | 0.7658 | 54.95 min |

**Difference:** E1 − E0 = **+2.78 percentage points** on validation accuracy, on one seed. The run time measured here is the sum of `epoch_seconds`; total SLURM job duration may be longer. Do not call this a statistical significance result or a test-set result.

**Primary evidence:** `logs/e0_final_s42.csv`, `logs/e1_final_s42.csv`, `checkpoints/e0_final_s42/best.pth`, `checkpoints/e1_final_s42/best.pth`, run config/environment files and corresponding SLURM outputs. Preserve originals, split indices, and a versioned source snapshot.

## 3. Proposed freeze for E2/E3 integration (to be checked against repository's detailed protocol)

**Keep unchanged:** dataset and split indices; transformations and normalization order; ResNet-18; 200 epochs; SGD+cosine; batch=128; seed=42; E1 CutMix alpha=1.0, p=0.5; checkpoint selection by val Top-1. No validation/test CutMix.

**Candidate construction (README guidance):** sample source/target image pairing and E1 base rectangle once per sample using shared rule; form K candidates by including base rectangle as candidate 0 and translating the clipped rectangle without changing its actual height/width. For `random` choose candidate 0; for `dual_edge` and `seam_aware` score the same candidates; use source crop at same coordinates in target. Freeze tie-breaking, boundary handling and RNG behavior across policies.

**Initial experiment design (A proposals, NOT frozen method defaults):**

| Comparison | Candidate parameter settings | Interpretation |
|---|---|---|
| E2-S vs E2 | Source-only (`gamma=0`) vs dual-side (`gamma=0.5`) | Content balance; verify gamma=0 equivalence |
| Candidate count | `K = 1, 4, 8` | Candidate selection benefit vs overhead |
| E3 seam weight | `beta = 0`, plus two nonzero values after B/C agree on score normalization | The correct numerical beta scale depends on actual normalized edge/seam scores; do **not** freeze arbitrary values before measuring scales |
| E4 later | `r=0` vs nonzero `r`, `mu` and area budget chosen with C/D | `r=0` must equal E3; match E4-R geometric constraints |

**Required agreement with B/C before formal E2/E3 jobs:** exact candidate construction; edge normalization and `eps`; seam normalization and beta scale; source-index pairing and reproducible RNG; metadata schema, mask shape (`N×1×H×W` boolean), box convention `(top,left,bottom,right)` half-open; precision and error/fallback rules. If the original E1 sampler needs changes to match the shared candidate scheme, rebuild affected baselines at the same fixed 200-epoch protocol rather than comparing incompatible implementations.

## 4. Augmentation API and required tests

```python
cutmix_batch(images, labels, policy, alpha, p, k, gamma,
             beta=0.0, seam_radius=0, seam_mu=0.0)
# -> mixed_images, labels_a, labels_b, lam_actual, metadata
```

- Inputs: images `[N,3,32,32]` on source device, pixel values `[0,1]`; labels `[N]`.
- Outputs: `mixed_images` same shape/device; labels_a/b `[N]`; `lam_actual` `[N]`, finite and within `[0,1]`; metadata containing final bool mask, sampled base box, source pairing, optional candidate scores and diagnostics.
- Source pixels must originate from the original unmodified batch.
- Check label weight against final mask area, exact pixel replacement, device/dtype, K=1, empty/full or clipped boxes, constant images, boundary conditions, and no NaN.
- Degeneracy tests: `gamma=0` dual_edge ≡ source_edge; `beta=0` seam_aware ≡ dual_edge; `r=0` seam_dp ≡ seam_aware under identical randomness.
- Timing must include candidate/edge/seam scoring, mask creation and mixing; warm up and synchronize CUDA as needed. Report training end-to-end runtime separately.

## 5. Planned runs and ownership

| Stage | Owner | Deliverable / acceptance |
|---|---|---|
| E0/E1 | A | Completed seed 42, best checkpoint + raw CSV (done) |
| E2-S / E2 | B algorithm, A integration | Unit tests; K=1/4/8; 1-batch smoke; 5–10 epoch speed pilot; 200-epoch run |
| E3 | C algorithm, A integration | Same candidate/pairing rules, seam cost tests, beta=0 check; speed pilot; 200-epoch run |
| E4/E4-R | C algorithm, A integration, D diagnostics | DP masks and random matched shape; r=0 check; area/perimeter/fallback matching; speed pilot then full run if feasible |
| Evaluation and figures | D | Clean val Top-1, macro-F1 on agreed selected models, efficiency, seam discontinuity and area/perimeter/fallback diagnostic tables |
| Multi-seed | A + D | Repeat all *main compared methods* at 2 additional seeds (e.g. 123,456) after method choices freeze; report raw seeds and mean±SD |
| Final test | D + A | Evaluate only finally selected checkpoint/methods on held-out CIFAR-100 official test; no test tuning |

## 6. Execution & reproduction checks for next phase

1. Preserve the current E0/E1 runs; do not overwrite existing CSVs and checkpoint directories.
2. Obtain B/C code or verify availability, agree on interface before integration.
3. Add policy registration in `augmentations/__init__.py`; ensure no policy reuses an E0 stub accidentally.
4. Run CPU unit tests; submit one GPU smoke test (2 batches × 2 epochs); then a 5–10 epoch speed pilot **with a fresh run name**. The existing baseline SLURM script accepts only `none|random` and needs an E2/E3-capable version.
5. Use a job time limit ≤ 05:30:00 after speed pilot shows expected total duration below that. Current E0/E1 time does not predict E4 DP overhead.
6. Preserve the per-run resolved config, environment, code snapshot, seed, split indices checksum, raw logs and checkpoints.
7. D should keep pilot and main results separate, and must report missing/negative results faithfully.

## 7. Open points to reconcile with repository and team

- Read the actual `docs/experiment_protocol.md` and `augmentations/README.md` before final freeze; this handoff used the supplied project README as its source.
- Freeze E2 edge normalizer and gamma, E3 seam normalizer / beta sweep, and exact candidate generation & per-sample RNG rules with B/C.
- Decide whether E0/E1 are comparable to E2/E3 candidate-generation mechanism as implemented; if not, rerun after standardization.
- Decide hyperparameter search budget equal across methods, seed plan, macro-F1 implementation and inference-only robustness tests.
- Verify reported raw source code provenance because GPU machine was used without Git working tree.
