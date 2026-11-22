# Implemented experiment protocol

This document records the code and completed experiments in the integrated submission. It supersedes the development draft. The original assignment brief remains archived in [project_brief.md](project_brief.md). Commands are in [the README](../README.md) and [running guide](running.md).

## Dataset and model

- CIFAR-100: the official 50,000-image training partition is divided into 45,000 training and 5,000 validation images using the committed seed-42 stratified indices. Each class contributes 450/50 images. The split is shared by every experiment.
- Training images: random 32×32 crop with padding 4, random horizontal flip, tensor conversion to RGB [0,1], optional mixing, then channel normalization. Validation uses clean tensor conversion and the same normalization, without mixing or random crop/flip.
- Mean `(0.5071, 0.4867, 0.4408)`, standard deviation `(0.2675, 0.2565, 0.2761)`.
- Four-stage ResNet-18 `[2,2,2,2]`, 3×3 stride-1 stem, no initial max-pool, 100-class head. It is a CIFAR adaptation of torchvision ResNet-18, not the upstream three-stage CIFAR ResNet.
- All reported accuracies are validation scores. The official 10,000-image test partition has not been evaluated.

## Training and selection

One `train.py` loop handles `none`, `random`, `source_edge`, `dual_edge`, `seam_aware`, `seam_dp`, and `seam_random`. Formal runs use 200 epochs, batch 128, SGD with learning rate 0.1, momentum 0.9, weight decay 0.0005 and cosine decay. Mixing uses alpha=1 and per-example probability p=0.5. Short smoke configurations are separate and never enter the formal result tables.

The best checkpoint is selected by maximum clean validation Top-1; strict improvement keeps the earliest epoch on a tie. The final checkpoint is also saved. We report the final 10-epoch mean as a secondary summary, without treating correlated epochs as independent runs. Configuration, environment, split hashes, source hashes and CSV logs accompany the recorded runs. New output names cannot overwrite existing runs. Resume training is not implemented.

The method configurations record the values actually used, rather than claiming globally optimal parameters. Ablations and exploratory choices are identified in the result tables. Official test scores were not used for selection.

## Shared masks, labels and sampling

For target image A, source image B, and a binary mask M=1 where B is used:

`mixed = (1 - M) * A + M * B`

The target label weight is `lambda_actual = 1 - sum(M)/(H*W)`. The loss is the per-example weighted sum of the two cross-entropies, averaged over the batch. Sources are always read from the unmodified input batch. Coordinates `(top,left,bottom,right)` are half-open.

The E1 sampler retains A's implementation of the original CutMix center-and-clip rule. Sampling uses a torch permutation, per-example mixing decisions, numpy Beta draws, and integer rectangle coordinates. Self-pairing and repeated proposals are allowed. The source and target use the same paste coordinates.

Candidate zero is the E1 rectangle. E2/E3 generate K−1 translations with the same actual width and height, so E1–E3 share the rectangle area. Candidate sampling uses an independent clone of the device RNG state and does not advance the global stream. K=4 candidates are a prefix of K=8 under the same seed and device. Ties select the first candidate. CPU and CUDA random streams are not assumed bitwise identical.

## Policies and parameters

| ID | Policy | Implemented behavior |
|---|---|---|
| E0 | none | No image mixing |
| E1 | random | Original random rectangular paste |
| E2-S | source_edge | Maximize normalized source Sobel content |
| E2 | dual_edge | Maximize source content minus gamma times target content |
| E3 | seam_aware | E2 score minus beta times normalized RGB seam cost |
| E4 | seam_dp | E3 rectangle followed by constrained joint boundary DP |
| E4-R | seam_random | Same E3 rectangle and geometric rules, with random DP costs |

Main edge/seam configurations use K=4 and gamma=0.5; E2-S uses gamma=0. E3/E4 use beta=1. E4 uses radius=2, bend penalty mu=0.05, minimum width=2 and relative area budget=0.2. E4-R uses its training-image shape-calibrated bend penalty 0.225. B's K=1/4/8 and C's beta/radius/mu ablations are preserved in separate configuration files.

### Edge and seam scores

Sobel operates on grayscale `0.299R+0.587G+0.114B`, with replicate padding and gradient magnitude. Each edge map is normalized by its image mean, clamped at 1e-6. Constant inputs produce zero edges. Region means use float64 integral accumulation; scores retain the chosen working precision.

The raw seam cost is the mean channel-wise absolute RGB difference over all in-image, four-neighbor pixel pairs crossing the final mask. Values are calculated before model normalization. Image exterior is not an extra target pixel. Empty/full masks with no internal seam are counted as not applicable in reporting.

E3 normalizes the rectangle seam cost by the mean intrinsic neighbor difference of the source and target images, with an RGB-scale floor of 1e-3. This is image-level normalization, not a separate rescaling per candidate. Its score is `source_edge - gamma*target_edge - beta*normalized_seam`.

### Boundary search and random control

E4 holds top and bottom fixed and jointly optimizes the left/right row offsets within the radius band. Each path moves at most one pixel between adjacent rows. Joint states enforce minimum width and prevent intersection; bending is penalized. This solves the defined vertical-edge proxy, not a global optimum of the full final boundary.

The proposed final area must satisfy `abs(final_area - base_area) <= area_budget * base_area`; otherwise the method returns the E3 rectangle. Status codes record accepted, no patch, too small, infeasible and area fallback cases. Full final-mask seam cost and label weights are recomputed after this decision.

E4-R starts at the same E3 rectangle, uses the same feasible geometry and fallback rules, and replaces the image costs with random costs. Its bend penalty was calibrated on training-image shape statistics, without classifier/validation labels. Equal constraints do not imply identical area, perimeter or fallback distributions; the measured residual differences are retained in the diagnostics.

## Measurements and completed evidence

- **21 completed formal runs:** 18 original seed-42 runs across the labeled A/B/C environments, plus 3 seed-43 E1/E4/E4-R runs on one M5 Max host. Each contains 200 epochs and 45,000 training samples per epoch. Distinct hosts/backends prevent interpreting them as a controlled two-seed study.
- **Boundary audit:** 2,560 paired training inputs, 20 batches of 128, forced mixing p=1. RGB jumps are aggregated by eligible boundary-pair counts. Crop-padding sensitivity uses a transformed validity channel; black color alone is not a padding detector. Area, boundary length, bending, edge attachment and fallbacks are also recorded.
- **Paired sensitivity:** batch bootstrap over the raw paired audit, 4,000 draws, with padding, newly attached image-edge and observational shape-subset analyses. These intervals concern augmentation measurements, not classification or training-seed uncertainty.
- **Efficiency:** synchronized augmentation-only timings with warmup, and separately full training-process time. Host, batch size, repetition counts and timing scope accompany the source files. No inference-time model is added.
- **Accuracy/F1:** original logs and checkpoint re-evaluation records are retained. New same-host checkpoints were independently reloaded on CPU and matched the saved best-validation scores. Public predictions permit recomputing confirmation Top-1 and Macro-F1 without distributing the weights.

The evidence supports lower average RGB seam discontinuity and classification near the CutMix baseline, rather than a stable classification improvement. Complete source data and scope are in [RESULTS_D.md](../RESULTS_D.md) and [RESULTS_AUDIT.md](../RESULTS_AUDIT.md).

## Checks and code provenance

Correctness tests cover mask/pixel/label consistency, beta=0 and radius=0 reductions, K=1 equivalence, shared sampling/RNG behavior, a slow seam reference, brute-force DP comparisons, geometry/fallback constraints, checkpoint safety and the shared training/evaluation path. Accelerator-specific checks skip explicitly when hardware is absent.

The original CutMix snapshot remains byte-for-byte unchanged at commit `2d8eb68faff7fe4962776ad51d175c3b01a25734`. Its [manifest](../third_party/cutmix_provenance.json), LICENSE and NOTICE are preserved. [NOTICE.md](../NOTICE.md) maps adaptations and attribution. The original A handoff remains archived; the top-level code is the integrated implementation.
