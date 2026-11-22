# Result figures

`d_validation_curves.png` and `d_efficiency.png` come from the original formal logs and recorded synchronized timings. `d_boundary_distributions.png` comes from D's raw paired audit. `d_mixed_examples.png` contains actual mixed training images. B/C retain their additional figures in their own result directories.

`python analysis/reproduce_results.py --out-dir outputs/reproduced` regenerates the numeric-result figures in a separate output directory, together with the confirmation curves. Regenerating mixed-image examples instead requires the actual dataset. [All source evidence](../README.md) · [Running guide](../../docs/running.md).
