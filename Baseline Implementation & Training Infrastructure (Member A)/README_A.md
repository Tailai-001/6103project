\# AI6103 — Member A Handoff



\## 1. Responsibility



Member A is responsible for the baseline implementation, unified training infrastructure, GPU training, and integration of E0–E4.



\## 2. Completed Work



\- CIFAR-100 preparation and fixed stratified train/validation split (45,000 / 5,000).

\- Four-stage CIFAR-adapted ResNet-18.

\- Unified training and validation pipeline.

\- Original CutMix (E1).

\- GPU training on NVIDIA A40.

\- E0 and E1 formal experiments (200 epochs, seed 42).



\## 3. Baseline Results



| Experiment | Best Val Accuracy | Best Epoch | Training Time |

|---|---:|---:|---:|

| E0 — No CutMix | 77.74% | 186 | 36.10 min |

| E1 — Original CutMix | 80.52% | 200 | 54.95 min |



Improvement: +2.78 percentage points.



These are single-seed validation results, not final test-set results.



\## 4. Training Configuration



\- Dataset: CIFAR-100

\- Model: CIFAR ResNet-18

\- Batch size: 128

\- Epochs: 200

\- Optimizer: SGD

\- Learning rate: 0.1

\- Momentum: 0.9

\- Weight decay: 0.0005

\- Scheduler: Cosine Annealing

\- CutMix alpha: 1.0

\- CutMix probability: 0.5

\- Main random seed: 42

\- Checkpoint selection: Best validation Top-1 accuracy



\## 5. Augmentation Interface



```python

cutmix\_batch(

&#x20;   images, labels, policy,

&#x20;   alpha=1.0, p=0.5, k=4,

&#x20;   gamma=0.5, beta=0.0,

&#x20;   seam\_radius=0, seam\_mu=0.0

)

```



Expected output:



```python

mixed\_images, labels\_a, labels\_b, lam\_actual, metadata

```



Conventions:



\- Images: `\[N,3,32,32]`, RGB in `\[0,1]`, before normalization.

\- Labels: `\[N]`.

\- Mask: `\[N,1,32,32]`, where 1 denotes Source pixels.

\- Rectangle coordinates: `(top,left,bottom,right)`, half-open.

\- `lam\_actual`: per-sample Target-label weight computed from the final Mask area.

\- Source pixels must come from the unmodified original batch.



E1–E3 should share the agreed base-rectangle and candidate-generation rules. Verify that the current implementation complies before final comparisons.



\## 6. Next Steps



\*\*Member B:\*\* Implement Sobel edge scoring, candidate rectangle generation, source-guided E2-S and dual-sided E2.



\*\*Member C:\*\* Implement E3 seam-aware rectangle selection, E4 dynamic-programming boundary optimization, and E4-R random boundary control.



\*\*Member A:\*\* Integrate B/C modules into the unified training pipeline, validate interfaces, and run common training experiments.



\*\*Member D:\*\* Evaluate classification accuracy, boundary metrics, efficiency, ablations, and multi-seed results.



\## 7. Reproducibility



Refer to `requirements-lock.txt`, `experiment\_protocol.md`, `configs/main.yaml`, and the saved train/validation split indices.



The training dataset and large model checkpoints are not included in the source-code handoff.

