"""Paired sensitivity analysis of D's existing training-image boundary audit.

No classifier, checkpoint, test images or new parameter search is involved.
Bootstrap units are the original 128-image batches, preserving donor/permutation
dependence within each batch. Intervals concern this augmentation audit, not
training-seed variability or classification accuracy.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHAPE_AREA_TOLERANCE = 0.02  # absolute image-area ratio, fixed before analysis
SHAPE_LENGTH_RATIO = 1.10
SHAPE_BEND_TOLERANCE = 2


def read_samples(path):
    samples = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            sample = int(row["sample"])
            arm = row["arm"]
            if arm in samples.setdefault(sample, {}):
                raise ValueError(f"Duplicate sample/arm: {sample}/{arm}")
            numeric = {key: float(value) for key, value in row.items()
                       if key not in ("arm", "status") and value != ""}
            if not all(np.isfinite(value) for value in numeric.values()):
                raise ValueError("Non-finite raw audit values")
            for key, value in numeric.items():
                if key.startswith("count_") and (value < 0 or value != int(value)):
                    raise ValueError("Invalid boundary count")
            samples[sample][arm] = dict(numeric, status=row.get("status", ""))
    if not samples or sorted(samples) != list(range(len(samples))):
        raise ValueError("Expected consecutive paired sample IDs starting at zero")
    expected = {"E1", "E2", "E3", "E4", "E4-R"}
    if any(set(arms) != expected for arms in samples.values()):
        raise ValueError("Every sample must contain all five arms")
    return [samples[i] for i in sorted(samples)]


def bootstrap_difference(a_sum, a_count, b_sum, b_count, clusters, repeats=4000, seed=6103):
    """Candidate-minus-reference difference of pair-weighted RGB ratios."""
    a_sum, a_count, b_sum, b_count = (np.asarray(x, dtype=float)
                                     for x in (a_sum, a_count, b_sum, b_count))
    clusters = np.asarray(clusters)
    if not len(a_sum) or not (a_count.sum() > 0 and b_count.sum() > 0):
        raise ValueError("Both arms need eligible boundary pairs")
    if any(len(x) != len(a_sum) for x in (a_count, b_sum, b_count, clusters)):
        raise ValueError("Paired arrays must have matching lengths")
    unique, inverse = np.unique(clusters, return_inverse=True)
    totals = np.stack([np.bincount(inverse, weights=x, minlength=len(unique))
                       for x in (a_sum, a_count, b_sum, b_count)], axis=1)
    point = a_sum.sum() / a_count.sum() - b_sum.sum() / b_count.sum()
    if len(unique) < 2:
        return float(point), None
    draw = np.random.default_rng(seed).integers(len(unique), size=(repeats, len(unique)))
    resampled = totals[draw].sum(axis=1)
    valid = (resampled[:, 1] > 0) & (resampled[:, 3] > 0)
    delta = (resampled[valid, 0] / resampled[valid, 1]
             - resampled[valid, 2] / resampled[valid, 3])
    return float(point), np.quantile(delta, [0.025, 0.975]).tolist()


def eligible_pair(a, b, variant, subset):
    if min(a[f"count_{variant}"], b[f"count_{variant}"]) <= 0:
        return False
    if subset in ("no_new_image_edge", "accepted_shape_close"):
        if a.get("touched_edge", 0) or b.get("touched_edge", 0):
            return False
    if subset == "accepted_shape_close":
        if a.get("status") != "accepted" or b.get("status") != "accepted":
            return False
        small = min(a["count_all"], b["count_all"])
        return (abs(a["area_ratio"] - b["area_ratio"]) <= SHAPE_AREA_TOLERANCE
                and max(a["count_all"], b["count_all"]) / small <= SHAPE_LENGTH_RATIO
                and abs(a["bends"] - b["bends"]) <= SHAPE_BEND_TOLERANCE)
    return True


def compare(samples, candidate, reference, variant, subset, batch_size, repeats):
    kept = [(i, item[candidate], item[reference]) for i, item in enumerate(samples)
            if eligible_pair(item[candidate], item[reference], variant, subset)]
    if not kept:
        return dict(candidate=candidate, reference=reference, variant=variant,
                    subset=subset, paired_samples=0, difference=None, ci95=None)
    ids, a, b = zip(*kept)
    sa = np.array([r[f"sum_{variant}"] for r in a])
    ca = np.array([r[f"count_{variant}"] for r in a])
    sb = np.array([r[f"sum_{variant}"] for r in b])
    cb = np.array([r[f"count_{variant}"] for r in b])
    delta, ci = bootstrap_difference(sa, ca, sb, cb, np.array(ids) // batch_size, repeats)
    av, bv = sa.sum() / ca.sum(), sb.sum() / cb.sum()
    per_image = sa / ca - sb / cb
    return dict(candidate=candidate, reference=reference, variant=variant, subset=subset,
                paired_samples=len(kept), retained_fraction=len(kept) / len(samples),
                clusters=len(np.unique(np.array(ids)//batch_size)), candidate_rgb=float(av),
                reference_rgb=float(bv), difference=delta, ci95=ci,
                relative_reduction_percent=float(100*(bv-av)/bv) if bv else None,
                lower_rgb_fraction=float((per_image < -1e-12).mean()),
                equal_rgb_fraction=float((np.abs(per_image) <= 1e-12).mean()),
                mean_image_difference=float(per_image.mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=ROOT/'results/member_d/boundary_samples.csv')
    parser.add_argument("--out", type=Path, default=ROOT/'results/local_audit')
    parser.add_argument("--bootstrap", type=int, default=4000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("Need at least 100 bootstrap draws")
    samples = read_samples(args.samples)
    source = json.loads((ROOT/'results/member_d/boundary_summary.json').read_text())
    n_batch, batch_size = source["batches"], source["batch_size"]
    if len(samples) != n_batch*batch_size:
        raise ValueError("Audit sample count differs from declared complete batches")
    rows = []
    for candidate, reference in [("E3", "E2"), ("E4", "E3"), ("E4", "E4-R")]:
        for variant in ("all", "without_any_padding"):
            subsets = ["all_valid", "no_new_image_edge"]
            if reference == "E4-R":
                subsets.append("accepted_shape_close")
            for subset in subsets:
                rows.append(compare(samples, candidate, reference, variant, subset,
                                    batch_size, args.bootstrap))
    changed = [(s['E4'], s['E3']) for s in samples
               if s['E4'].get('status')=='accepted' and s['E4'].get('changed')
               and min(s['E4']['count_all'],s['E3']['count_all'])>0]
    worsened = sum(a['sum_all']/a['count_all'] > b['sum_all']/b['count_all']+1e-12
                  for a,b in changed)
    result = dict(source_sha256=hashlib.sha256(args.samples.read_bytes()).hexdigest(),
                  source_code_base='8408c039b0ad33b97ffc8f9214135f024ef71efb',
                  scope='existing forced-mix training-image audit; no classification inference',
                  batches=n_batch,batch_size=batch_size,paired_inputs=len(samples),
                  bootstrap_draws=args.bootstrap,bootstrap_seed=6103,
                  resampling_unit='paired original batches (donor dependence retained)',
                  matching=dict(area_ratio_absolute=SHAPE_AREA_TOLERANCE,
                                max_length_ratio=SHAPE_LENGTH_RATIO,max_bend_difference=SHAPE_BEND_TOLERANCE,
                                exclude_new_image_edge=True,require_both_accepted=True),
                  dp_full_cost_increase=dict(changed_accepted_samples=len(changed),
                                            worsened_samples=worsened,
                                            fraction=worsened/len(changed) if changed else None),
                  comparisons=rows)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'paired_boundary_evidence.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    flat=[{**{k:v for k,v in row.items() if k!='ci95'},
           'ci95_low':row['ci95'][0] if row['ci95'] else None,
           'ci95_high':row['ci95'][1] if row['ci95'] else None} for row in rows]
    with (args.out/'paired_boundary_evidence.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(flat)
    lines=['# Paired boundary sensitivity results','',
           'Negative differences favour the candidate. 95% intervals resample the original paired batches.',
           'These intervals describe augmentation behaviour, not seed-level classification uncertainty.',
           'Shape-close is a fixed observational subset, not an exactly shape-matched causal control.','',
           '| Comparison | Boundary pairs | Subset | N paired | Candidate − reference | 95% batch bootstrap CI |',
           '|---|---|---|---:|---:|---|']
    for row in rows:
        ci=row['ci95']
        lines.append(f'| {row["candidate"]} − {row["reference"]} | {row["variant"]} | {row["subset"]} | '
                     f'{row["paired_samples"]} | {row["difference"]:.5f} | [{ci[0]:.5f}, {ci[1]:.5f}] |'
                     if ci else f'| {row["candidate"]} − {row["reference"]} | {row["variant"]} | {row["subset"]} | {row["paired_samples"]} | N/A | N/A |')
    lines += ['',f'Full RGB discontinuity increased for {worsened}/{len(changed)} accepted changed E4 masks.',
              'The local DP objective therefore does not guarantee improvement of the complete final boundary metric.',
              'The source is a 20-batch training audit with p=1; do not extrapolate these intervals to p=0.5 training accuracy.']
    (args.out/'paired_boundary_evidence.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
