# Correctness and integration tests

Install `requirements-dev.txt` and run **`python -m pytest -q`** from the root. `pytest.ini` restricts collection to this directory, so the unchanged upstream reference is not imported as a test suite. No dataset is required.

Coverage includes Sobel/integral-image references, candidate areas and RNG rules, E1/K=1 and gamma=0 equivalence, mask/pixel/label consistency, slow seam references, beta=0 and radius=0 reductions, brute-force checks of joint DP, shape/area/fallback constraints, random-boundary reproducibility, checkpoint safety, evidence validation, and multi-step shared-loop CPU/MPS comparisons.

Hardware-specific CUDA/MPS tests skip when unavailable. `unittest discover` collects only the unittest-style subset and is not the complete acceptance command. [Actual submission checks](../docs/submission_validation.md) record the environment and counts.

The separate `python scripts/smoke_test.py` command exercises every policy with a full ResNet-18 optimizer step, clean evaluation and a temporary checkpoint roundtrip. Add `--data-root` to use an existing CIFAR-100 archive; no automatic download occurs.
