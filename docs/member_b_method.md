# Member B: implementation notes for the report

## Edge-guided rectangle selection

Given a target image A and a paired source B, we generate K rectangles with equal actual height and width. Candidate 0 is the clipped rectangle sampled by Member A's Original CutMix implementation. The remaining candidates translate that rectangle to uniformly sampled valid positions. Each candidate uses the same coordinates in A and B. This controls replacement area while changing which content is replaced and inserted.

We compute grayscale intensity as 0.299R + 0.587G + 0.114B, followed by unscaled 3×3 Sobel derivatives with replicated boundary padding. The gradient magnitude is normalized by its image-wide mean, with a denominator floor of 1e-6. Constant images produce zero scores. Region means use double-precision integral accumulation to avoid cancellation in tiny low-gradient crops, then return to the edge-map dtype; candidates are never independently normalized.

The source-only ablation (E2-S) maximizes the mean normalized source gradient magnitude inside the rectangle. The dual-sided policy (E2) maximizes source mean minus gamma times target mean, with the predeclared default gamma=0.5. Ties select the first candidate. This is a low-level structural heuristic: gradient magnitude is not a semantic saliency estimate, and neither a higher score nor a lower target score guarantees better classification.

We construct a Boolean source mask M and form `(1−M)A + MB`, reading B from the original unmodified batch. Target-label weight is `1−sum(M)/(HW)` for each sample. Empty regions leave the image unchanged. All candidates have the same area as the base rectangle, so the label weight is invariant to rectangle selection. Original image pairing, per-sample CutMix activation, numpy Beta draws, integer sizes, center sampling and clipping are preserved from A.

Candidate translation draws use a cloned torch RNG state and a fixed N×2 draw for each added candidate. They do not advance the global RNG. Candidate sets therefore have matching prefixes across K values on the same device; changing K or scoring policy does not alter subsequent base rectangles or pairings. Separate CPU and CUDA generators need not produce the same samples across devices.

## Implementation and cost

Sobel, image-level normalization, integral-image construction, rectangle scoring and mask composition use batched PyTorch operations on the input device. The scoring cost is O(NHW+NK); candidate metadata requires O(NK) storage, while image maps and masks require O(NHW). Candidate drawing loops over K, not batch samples. We retain A's per-sample sampler, including its host synchronization, to preserve the baseline exactly. The compatibility implementation also discards the temporary mixed image produced by A's sampler. Consequently, asymptotic scoring cost alone does not describe complete augmentation overhead; synchronized end-to-end augmentation timing is reported separately.

## Validation and experiment interpretation

Correctness checks include analytical Sobel ramps, constant inputs, integral-image means against direct crop means, exact E1 equivalence at K=1, source-only equivalence at gamma=0, immutable source pixels, final mask/label area agreement, equal-area candidate geometry, bounds, empty/full masks, first-index ties, seed reproduction, K-prefix stability, singleton/tiny inputs, floating dtypes and CPU/CUDA execution. A controlled synthetic pair verifies that the source-only method favors a stronger source edge while a sufficiently large target penalty changes selection to preserve target structure.

All B configurations use A's saved 45,000/5,000 split, seed 42, CIFAR-adapted four-stage ResNet-18, 200 epochs, batch size 128, SGD (initial learning rate 0.1, momentum 0.9, weight decay 0.0005), cosine annealing, alpha=1 and mixing probability 0.5. The earliest maximum clean validation Top-1 selects the checkpoint. K=1/4/8 are evaluated for dual-sided selection, with source-only K=4/8 and a shared K=1 control. No gamma search or official test evaluation is performed for B. Raw CSVs and checkpoint re-evaluation determine actual results; this method note asserts no improvement in advance.

A's uploaded E0/E1 validation results were obtained on different hardware/software. They provide project context, while B's K=1 control checks baseline behavior in B's environment. Report single-seed results without a fabricated standard deviation or statistical significance claim. Member D/A should coordinate additional seeds and final held-out test evaluation. Member C should reuse the same candidates and normalized source/target scores when adding seam costs.

## Measured seed-42 findings

All five 200-epoch runs completed. Best clean validation Top-1 was 80.06% for the K=1 control, 79.64%/79.62% for dual-sided K=4/8, and 80.36%/79.30% for source-only K=4/8. Source-only K=4 was 0.30 percentage points above the local control; the default dual-sided configurations did not improve it. This single-seed result does not establish statistical significance or test-set improvement. Increasing K was not beneficial here. All five saved best checkpoints reproduced their logged validation accuracy; detailed raw evidence is in results/member_b/.
