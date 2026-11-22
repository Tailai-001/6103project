# Data and the fixed split

`dataset.py` is the integrated A loader: 45,000/5,000 images from the official CIFAR-100 training partition, random crop/flip only for training, and RGB tensors in [0,1]. Model normalization happens after mixing in the shared trainer. Validation has no mixing or random transform.

Run `python scripts/prepare_data.py --download` from the root to explicitly obtain the official archive and verify split hashes and per-class counts. An existing archive can be checked with `--data-root /path/to/cifar100`; without `--download`, nothing is downloaded. Raw images remain excluded from Git.

`split.py` now validates the supplied indices without rewriting them and needs no scikit-learn. Run `python -m data.split` to check them without a dataset. The original generation script remains unchanged in the A archive. [Hashes and split details](splits/README.md) · [Complete running guide](../docs/running.md).

The official test partition has not been used for the reported measurements.
