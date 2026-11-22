# Code and research attribution

This AI6103 project extends **CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features**, Yun et al., ICCV 2019. [Paper](https://arxiv.org/abs/1905.04899) · [Official implementation](https://github.com/clovaai/CutMix-PyTorch).

The original source snapshot is stored without changes in `third_party/CutMix-PyTorch/`, pinned to commit `2d8eb68faff7fe4962776ad51d175c3b01a25734`. Its [MIT LICENSE](third_party/CutMix-PyTorch/LICENSE) and [NOTICE](third_party/CutMix-PyTorch/NOTICE), including notices for incorporated components, remain part of this distribution. File hashes are recorded in [cutmix_provenance.json](third_party/cutmix_provenance.json).

The original A handoff is preserved in `Baseline Implementation & Training Infrastructure (Member A)/`. The integrated `train.py`, `models/resnet.py`, `data/dataset.py` and `augmentations/cutmix.py` derive from that handoff; the baseline mixing/loss design follows CutMix. The model adapts torchvision ResNet-18 to CIFAR images. The current `data/split.py` validates the supplied fixed indices; the original split-generation script remains in the A archive.

Project additions outside the upstream snapshot include edge-guided candidate selection, RGB seam scoring, constrained joint boundary dynamic programming, its random-cost control, unified evaluation, diagnostics and evidence-reproduction tools. Boundary optimization is inspired by image quilting/stitching and seam-path ideas; it does not perform image resizing or soft blending. Paper notes and verified bibliographic entries are in `docs/literature/` and `report/references.bib`.

Third-party licenses continue to govern their respective code. This notice records provenance and does not replace the upstream licenses or claim exclusive ownership of pre-existing methods.
