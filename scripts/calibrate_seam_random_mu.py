"""Calibrate E4-R's bend penalty (seam_random_mu) to match E4's shapes.

Member C. Shape statistics only: no model, no labels, no validation data.
For every batch the RNG is reseeded identically before E4 and before each
E4-R candidate, so all arms reshape exactly the same E3 rectangles and only
the boundary differs. The chosen value minimises the gap in mean bends
(total pixels of edge shift between rows, fallbacks counting 0) over all
samples that received a patch; boundary length, area change and acceptance
are reported so the residual mismatch can be stated in the report.

p is forced to 1 here (shape given a patch does not depend on p); every
other augmentation parameter comes from the E4 config / flags.

Examples
  # On the training server (CIFAR-100 already under data/cifar100):
  python scripts/calibrate_seam_random_mu.py --config configs/c_seam_dp.yaml \
      --batches 40 --out results/member_c/seam_random_mu_calibration.json
  # Local smoke test without the dataset:
  python scripts/calibrate_seam_random_mu.py --synthetic --batches 4
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from augmentations import cutmix_batch  # noqa: E402
from augmentations.seam_dp import STATUS_ACCEPTED, STATUS_AREA, STATUS_NO_PATCH  # noqa: E402

E4_KEYS = ("alpha", "k", "gamma", "beta", "seam_radius", "seam_mu",
           "seam_min_width", "seam_area_budget")
DEFAULTS = {"alpha": 1.0, "k": 4, "gamma": 0.5, "beta": 1.0, "seam_radius": 2,
            "seam_mu": 0.05, "seam_min_width": 2, "seam_area_budget": 0.1}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", help="YAML with an 'augmentation' block for E4")
    for key in E4_KEYS:
        ap.add_argument("--" + key.replace("_", "-"), type=float, default=None)
    ap.add_argument("--grid", default=",".join(f"{v:.2f}" for v in np.arange(0, 1.0001, 0.05)),
                    help="comma-separated candidate seam_random_mu values")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", default=str(ROOT / "data" / "cifar100"))
    ap.add_argument("--split-dir", default=str(ROOT / "data" / "splits"))
    ap.add_argument("--synthetic", action="store_true",
                    help="smooth random images instead of CIFAR (pipeline check only)")
    ap.add_argument("--out", help="optional JSON output path")
    return ap.parse_args()


def e4_params(args):
    params = dict(DEFAULTS)
    if args.config:
        import yaml
        with open(args.config) as f:
            aug = yaml.safe_load(f).get("augmentation", {})
        params.update({k: aug[k] for k in E4_KEYS if k in aug})
    for key in E4_KEYS:
        val = getattr(args, key)
        if val is not None:
            params[key] = val
    for key in ("k", "seam_radius", "seam_min_width"):
        params[key] = int(params[key])
    return params


def batches(args):
    if args.synthetic:
        g = torch.Generator().manual_seed(args.seed)
        for _ in range(args.batches):
            x = torch.nn.functional.interpolate(
                torch.rand(args.batch_size, 3, 8, 8, generator=g), size=32,
                mode="bilinear", align_corners=False)
            yield x, torch.zeros(args.batch_size, dtype=torch.long)
        return
    from data.dataset import build_dataloaders
    train_loader, _ = build_dataloaders(args.data_root, args.split_dir, args.batch_size,
                                        num_workers=0, seed=args.seed)[:2]
    for i, (x, y) in enumerate(train_loader):
        if i >= args.batches:
            break
        yield x, y


def collect(meta):
    has_patch = meta["dp_status"] != STATUS_NO_PATCH
    base = meta["area_base"].clamp_min(1).double()
    return {
        "bends": meta["bends"][has_patch].double(),
        "length": meta["seam_count"][has_patch].double(),
        "area_change": ((meta["area_final"] - meta["area_base"]).abs().double() / base)[has_patch],
        "accepted": (meta["dp_status"] == STATUS_ACCEPTED)[has_patch].double(),
        "changed": (meta["dp_changed"] & (meta["dp_status"] == STATUS_ACCEPTED))[has_patch].double(),
        "area_fallback": (meta["dp_status"] == STATUS_AREA)[has_patch].double(),
    }


def summarise(chunks):
    out = {}
    for key in chunks[0]:
        v = torch.cat([c[key] for c in chunks])
        out[key] = float(v.mean())
        if key in ("bends", "length"):
            out[key + "_p50"] = float(v.median())
            out[key + "_p90"] = float(v.quantile(0.9))
    out["n"] = int(torch.cat([c["bends"] for c in chunks]).numel())
    return out


def main():
    args = parse_args()
    params = e4_params(args)
    grid = [float(v) for v in args.grid.split(",") if v.strip()]
    if any(v < 0 for v in grid):
        raise SystemExit("grid values must be nonnegative")

    e4_chunks, r_chunks = [], {mu: [] for mu in grid}
    for b, (x, y) in enumerate(batches(args)):
        def run(policy, **extra):
            np.random.seed(args.seed + b)
            torch.manual_seed(args.seed + b)
            return cutmix_batch(x, y, policy=policy, p=1.0, **params, **extra)[4]
        ref = run("seam_dp")
        e4_chunks.append(collect(ref))
        for mu in grid:
            meta = run("seam_random", seam_random_mu=mu)
            assert torch.equal(meta["boxes"], ref["boxes"]), "E3 rectangles diverged"
            r_chunks[mu].append(collect(meta))
    if not e4_chunks:
        raise SystemExit("no batches read")

    e4 = summarise(e4_chunks)
    rows = {mu: summarise(r_chunks[mu]) for mu in grid}
    best = min(grid, key=lambda mu: (abs(rows[mu]["bends"] - e4["bends"]), mu))

    source = "synthetic smooth images" if args.synthetic else "CIFAR-100 train split (45k)"
    print(f"data: {source} | batches {len(e4_chunks)} x {args.batch_size} | samples with patch {e4['n']}")
    print("E4 params:", json.dumps(params))
    hdr = f"{'arm':16s} {'bends':>6s} {'p50':>5s} {'p90':>5s} {'length':>7s} {'|dA|/A':>7s} {'accept':>7s} {'changed':>8s} {'area-fb':>8s}"
    print(hdr)
    def line(name, s):
        print(f"{name:16s} {s['bends']:6.2f} {s['bends_p50']:5.1f} {s['bends_p90']:5.1f} {s['length']:7.1f} "
              f"{s['area_change']:7.3f} {s['accepted']:7.1%} {s['changed']:8.1%} {s['area_fallback']:8.1%}")
    line("E4 seam_dp", e4)
    for mu in grid:
        line(f"E4-R mu={mu:.3f}" + (" *" if mu == best else ""), rows[mu])
    gap = {k: rows[best][k] - e4[k] for k in ("bends", "length", "area_change", "accepted", "changed")}
    print(f"\nchosen seam_random_mu = {best:.3f} (closest mean bends)")
    print("residual E4-R minus E4: " + ", ".join(f"{k} {v:+.3f}" for k, v in gap.items()))

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "data": source, "batches": len(e4_chunks), "batch_size": args.batch_size,
            "seed": args.seed, "e4_params": params, "grid": grid,
            "criterion": "min |mean bends(E4-R) - mean bends(E4)| over samples with a patch",
            "chosen_seam_random_mu": best, "e4": e4,
            "e4r": {f"{mu:.4f}": rows[mu] for mu in grid}, "residual": gap,
        }, indent=2))
        print("wrote", out)


if __name__ == "__main__":
    main()
