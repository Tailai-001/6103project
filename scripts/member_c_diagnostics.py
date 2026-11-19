"""Member C diagnostics for E3 / E4 / E4-R on the CIFAR-100 TRAIN split.

Augmentation behaviour only: no model, no labels used, no validation or test
data. Reports E3 selection changes and seam cost vs E2, E4 acceptance /
fallback / seam change / border snapping, E4 vs E4-R shape statistics, and
how often seams run through RandomCrop's black padding.

  python scripts/member_c_diagnostics.py [--batches 20] [--random-mu 0.225]
"""
import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = str(pathlib.Path(__file__).resolve().parents[1])
_ap = argparse.ArgumentParser()
_ap.add_argument("--batches", type=int, default=20)
_ap.add_argument("--random-mu", type=float, default=0.225)
_ap.add_argument("--area-budget", type=float, default=0.2)
_args = _ap.parse_args()
N_BATCHES, RANDOM_MU = _args.batches, _args.random_mu
sys.path.insert(0, ROOT)
from augmentations import cutmix_batch  # noqa: E402
from data.dataset import build_dataloaders  # noqa: E402

torch.set_num_threads(4)
train_loader, _ = build_dataloaders(f"{ROOT}/data/cifar100", f"{ROOT}/data/splits", 128,
                                    num_workers=0, seed=42)
batches = []
for i, (x, y) in enumerate(train_loader):
    if i >= N_BATCHES:
        break
    batches.append((x, y))
print(f"CIFAR-100 train split: {len(batches)} batches x 128 = {len(batches) * 128} images\n")

E4 = dict(p=1.0, k=4, gamma=0.5, seam_radius=2, seam_mu=0.05,
          seam_min_width=2, seam_area_budget=_args.area_budget)


def run_all(policy, **kw):
    metas = []
    for b, (x, y) in enumerate(batches):
        np.random.seed(1000 + b)
        torch.manual_seed(1000 + b)
        metas.append(cutmix_batch(x, y, policy=policy, **kw)[4])
    return metas


def cat(metas, key):
    return torch.cat([m[key] for m in metas])


# ---------- E3 vs E2 ----------
print("== E3 seam_aware vs E2 dual_edge (K=4, gamma=0.5) ==")
e2 = run_all("dual_edge", p=1.0, k=4, gamma=0.5)
sel2 = cat(e2, "selected_indices")
print(f"{'beta':>5s} {'changed vs E2':>14s} {'D_hat selected':>15s} {'raw RGB seam':>13s}")
for beta in (0.0, 0.25, 0.5, 1.0, 2.0):
    e3 = run_all("seam_aware", p=1.0, k=4, gamma=0.5, beta=beta)
    valid = cat(e3, "seam_count") > 0
    ch = (cat(e3, "selected_indices") != sel2).float().mean()
    print(f"{beta:5.2f} {ch:14.1%} {cat(e3, 'seam_cost')[valid].mean():15.3f} "
          f"{cat(e3, 'seam_cost_raw')[valid].mean():13.4f}")

# ---------- E4 ----------
print("\n== E4 seam_dp (beta=1, r=2, mu=0.05, min_width=2) ==")
print(f"{'budget':>6s} {'accept':>7s} {'area-fb':>8s} {'changed':>8s} {'D_hat E3->E4 (changed)':>24s} "
      f"{'len E3->E4':>11s} {'bends':>6s} {'border snap':>12s}")
for budget in (0.05, 0.1, 0.2):
    m = run_all("seam_dp", **{**E4, "beta": 1.0, "seam_area_budget": budget})
    st, chg = cat(m, "dp_status"), cat(m, "dp_changed")
    has = st != 1
    ch = (st == 0) & chg
    boxes, mask = cat(m, "boxes"), torch.cat([mm["mask"][:, 0] for mm in m])
    snap = (((mask[:, :, 0].any(1)) & (boxes[:, 1] != 0)) |
            ((mask[:, :, -1].any(1)) & (boxes[:, 3] != 32))) & ch
    print(f"{budget:6.2f} {(st == 0)[has].float().mean():7.1%} {(st == 4)[has].float().mean():8.1%} "
          f"{ch[has].float().mean():8.1%} "
          f"{cat(m, 'seam_cost_base')[ch].mean():11.3f} -> {cat(m, 'seam_cost')[ch].mean():.3f} "
          f"{cat(m, 'seam_count_base')[ch].float().mean():5.1f}->{cat(m, 'seam_count')[ch].float().mean():4.1f} "
          f"{cat(m, 'bends')[ch].float().mean():6.2f} {snap.sum().item() / max(ch.sum().item(), 1):12.1%}")

# ---------- E4-R ----------
print(f"\n== E4 vs E4-R shape (budget {_args.area_budget}), E4-R mu = seam_mu vs calibrated {RANDOM_MU} ==")
e4 = run_all("seam_dp", **{**E4, "beta": 1.0})
rows = [("E4 seam_dp", e4),
        ("E4-R same mu", run_all("seam_random", **{**E4, "beta": 1.0})),
        (f"E4-R mu={RANDOM_MU}", run_all("seam_random", **{**E4, "beta": 1.0, "seam_random_mu": RANDOM_MU}))]
print(f"{'arm':16s} {'bends':>6s} {'length':>7s} {'|dA|/A':>7s} {'accept':>7s} {'D_hat':>6s}")
for name, m in rows:
    st = cat(m, "dp_status")
    has = st != 1
    dA = ((cat(m, "area_final") - cat(m, "area_base")).abs().float()
          / cat(m, "area_base").clamp_min(1).float())
    print(f"{name:16s} {cat(m, 'bends')[has].float().mean():6.2f} {cat(m, 'seam_count')[has].float().mean():7.1f} "
          f"{dA[has].mean():7.3f} {(st == 0)[has].float().mean():7.1%} {cat(m, 'seam_cost')[has].mean():6.3f}")

def run(policy, **kw):
    out = []
    for i, (x, y) in enumerate(batches):
        np.random.seed(1000 + i); torch.manual_seed(1000 + i)
        out.append((x, cutmix_batch(x, y, policy=policy, p=1.0, k=4, gamma=0.5, **kw)[4]))
    return out


print("\n== RandomCrop black padding (exact zeros) ==")
def pad_stats(runs):
    pairs = padpad = samples = samples_pad = 0
    for x, m in runs:
        src, tgt, mask = x[m["permutation"]], x, m["mask"]
        mixed = torch.where(mask, src, tgt)
        black = (mixed == 0).all(1, keepdim=True)          # exact zero = RandomCrop padding
        mk = mask
        sh = mk[..., 1:] != mk[..., :-1]; sv = mk[..., 1:, :] != mk[..., :-1, :]
        bh = black[..., 1:] & black[..., :-1]; bv = black[..., 1:, :] & black[..., :-1, :]
        n_pairs = sh.flatten(1).sum(1) + sv.flatten(1).sum(1)
        n_pad = (sh & bh).flatten(1).sum(1) + (sv & bv).flatten(1).sum(1)
        pairs += n_pairs.sum().item(); padpad += n_pad.sum().item()
        samples += (n_pairs > 0).sum().item(); samples_pad += ((n_pad > 0) & (n_pairs > 0)).sum().item()
    return padpad / pairs, samples_pad / samples
imgs = torch.cat([x for x, _ in batches])
print(f"images with any black padding pixel: {(imgs == 0).all(1).flatten(1).any(1).float().mean():.1%}")
print(f"{'policy':28s} {'seam pairs black|black':>23s} {'samples with such pairs':>24s}")
for name, pol, kw in [("E1 random", "random", {}), ("E2 dual_edge", "dual_edge", {}),
                      ("E3 seam_aware b=1", "seam_aware", {"beta": 1.0}),
                      ("E3 seam_aware b=2", "seam_aware", {"beta": 2.0}),
                      ("E4 seam_dp", "seam_dp", {"beta": 1.0, "seam_radius": 2, "seam_mu": 0.05, "seam_area_budget": _args.area_budget}),
                      (f"E4-R mu={RANDOM_MU}", "seam_random", {"beta": 1.0, "seam_radius": 2, "seam_mu": 0.05, "seam_area_budget": _args.area_budget, "seam_random_mu": RANDOM_MU})]:
    a, b = pad_stats(run(pol, **kw))
    print(f"{name:28s} {a:23.1%} {b:24.1%}")

def seam_excluding_pad(runs):
    tot_all = n_all = tot_np = n_np = 0.0
    for x, m in runs:
        src, tgt, mask = x[m["permutation"]], x, m["mask"]
        mixed = torch.where(mask, src, tgt)
        black = (mixed == 0).all(1, keepdim=True)
        dh = (mixed[..., 1:] - mixed[..., :-1]).abs().mean(1, keepdim=True)
        dv = (mixed[..., 1:, :] - mixed[..., :-1, :]).abs().mean(1, keepdim=True)
        sh = mask[..., 1:] != mask[..., :-1]; sv = mask[..., 1:, :] != mask[..., :-1, :]
        ph = black[..., 1:] & black[..., :-1]; pv = black[..., 1:, :] & black[..., :-1, :]
        tot_all += (dh * sh).sum().item() + (dv * sv).sum().item()
        n_all += sh.sum().item() + sv.sum().item()
        tot_np += (dh * (sh & ~ph)).sum().item() + (dv * (sv & ~pv)).sum().item()
        n_np += (sh & ~ph).sum().item() + (sv & ~pv).sum().item()
    return tot_all / n_all, tot_np / n_np
print(f"\n{'policy':28s} {'raw seam (all pairs)':>21s} {'raw seam (no black|black)':>26s}")
for name, pol, kw in [("E1 random", "random", {}), ("E2 dual_edge", "dual_edge", {}),
                      ("E3 seam_aware b=1", "seam_aware", {"beta": 1.0}),
                      ("E4 seam_dp", "seam_dp", {"beta": 1.0, "seam_radius": 2, "seam_mu": 0.05, "seam_area_budget": _args.area_budget}),
                      (f"E4-R mu={RANDOM_MU}", "seam_random", {"beta": 1.0, "seam_radius": 2, "seam_mu": 0.05, "seam_area_budget": _args.area_budget, "seam_random_mu": RANDOM_MU})]:
    a, b = seam_excluding_pad(run(pol, **kw))
    print(f"{name:28s} {a:21.4f} {b:26.4f}")
