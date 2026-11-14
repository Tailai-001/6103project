# Project guidance for coding assistants

This repository currently contains a project scaffold and an unchanged upstream reference snapshot. Read README.md, docs/experiment_protocol.md, and the relevant directory README before implementing work.

- Keep one training loop for `none`, `random`, `source_edge`, `dual_edge`, `seam_aware`, `seam_dp`, and `seam_random`. The README defines the updated experiment IDs; old E3 (dual-sided edge) is now E2.
- Use CIFAR-100 and a four-stage CIFAR-adapted ResNet-18 unless the team explicitly changes the protocol.
- Compute label weights from the final binary mask. Match rectangle area for E1–E3; control and report final area/shape for E4 and its random-boundary control. Preserve shared sampling rules, reproducible splits, and validation-only hyperparameter selection.
- Keep `third_party/CutMix-PyTorch/` byte-for-byte intact. Put adaptations in project modules and cite their source; retain applicable LICENSE and NOTICE text.
- Distinguish proposals, implemented behavior, and measured results. Never invent metrics or imply the scaffold has been trained.
- Do not download datasets, run long training, publish personal identifiers, or commit large artifacts unless requested for the current task.
- Choose validation appropriate to the change and document environment limitations. The initial scaffold does not require a GPU or PyTorch installation.
