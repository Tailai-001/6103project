# Result reproduction and evidence analysis

The portable entry point is **`python analysis/reproduce_results.py --out-dir outputs/reproduced`**, run from the root. It reads committed evidence, regenerates the original 18-run table, three local confirmation summaries, boundary/efficiency tables and four plots, and records input hashes. It neither trains nor requires datasets or checkpoints, and writes outside the evidence tree.

- `compile_results.py`: original A/B/C formal logs, settings, recorded checkpoint metrics, ablations, curves and synchronized costs. Its functions accept separate output directories.
- `plot_boundary.py`: D's raw paired audit CSV and summary, including validity-mask padding exclusions, final areas, boundary lengths and fallbacks.
- `audit_training_evidence.py`: checks the 18 original runs and their historical Git source hashes; requires full Git history.
- `paired_boundary_evidence.py`: paired batch bootstrap and padding/edge/shape sensitivity from the public CSV. These intervals do not measure classifier or seed uncertainty.
- `plot_paired_boundary_evidence.py`: plots those paired diagnostic intervals.
- `boundary_audit.py`: regenerates actual mixed-image diagnostics from an available dataset through the unified augmentation API.
- `compile_local_confirmation.py`: original checkpoint-based compilation of the completed same-host experiment. It requires its local checkpoints; the portable reproducer instead verifies the public logs and saved validation predictions.

[Integrated D analysis](../RESULTS_D.md) · [Independent audit](../RESULTS_AUDIT.md) · [Running guide](../docs/running.md).
