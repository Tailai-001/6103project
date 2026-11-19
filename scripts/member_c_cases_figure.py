"""Member C figure: E3 / E4 / E4-R masks side by side on CIFAR-100 TRAIN images.

  python scripts/member_c_cases_figure.py OUT.png [--random-mu 0.225]
Cases are picked automatically (largest E4 seam gain, largest E4-R vs E4 bend
gap, a typical sample, an E4 area fallback). Augmentation only, no model.
"""
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import argparse
import pathlib
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
_ap = argparse.ArgumentParser()
_ap.add_argument("out")
_ap.add_argument("--random-mu", type=float, default=0.225)
_ap.add_argument("--area-budget", type=float, default=0.2)
_args = _ap.parse_args()
sys.path.insert(0, ROOT)
from augmentations import cutmix_batch  # noqa: E402

OUT = _args.out
S = 7                      # upscale factor (32 px -> 224 px)
N = 512

from data.dataset import build_dataloaders
_loader, _ = build_dataloaders(f"{ROOT}/data/cifar100", f"{ROOT}/data/splits",
                               N, num_workers=0, seed=42)
x, y = next(iter(_loader))
CFG = dict(p=1, k=4, gamma=0.5, beta=1.0, seam_radius=2, seam_mu=0.05,
           seam_min_width=2, seam_area_budget=_args.area_budget)


def run(policy, **kw):
    np.random.seed(1)
    torch.manual_seed(1)
    return cutmix_batch(x, y, policy=policy, **{**CFG, **kw})


e3 = run("seam_aware")
e4 = run("seam_dp")
e4r = run("seam_random", seam_random_mu=_args.random_mu)
m3, m4, m4r = e3[4], e4[4], e4r[4]
perm = m3["permutation"]

# ---- pick cases ----
acc4 = (m4["dp_status"] == 0) & m4["dp_changed"]
acc4r = (m4r["dp_status"] == 0) & m4r["dp_changed"]
both = acc4 & acc4r & (m4["area_base"] > 120)
gain = m4["seam_cost_base"] - m4["seam_cost"]          # E4 improvement over E3
bend_gap = m4r["bends"] - m4["bends"]
cases = []
def pick(score, valid, label):
    s = torch.where(valid, score.float(), torch.tensor(-1e9))
    for idx in s.argsort(descending=True).tolist():
        if idx not in [c[0] for c in cases]:
            cases.append((idx, label))
            return
pick(gain, both, "A  E4 lowers the seam clearly")
pick(bend_gap, both, "B  E4-R much more jagged than E4")
med = both & (bend_gap - bend_gap[both].float().median()).abs().le(1)
pick(-(gain - gain[both].median()).abs(), med, "C  typical sample")
pick(m4["area_base"].float(), m4["dp_status"] == 4, "D  E4 area fallback -> same as E3")

# ---- drawing helpers ----
def to_img(t):
    a = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(a).resize((32 * S, 32 * S), Image.NEAREST)


def outline(draw, mask, color, width=2, off=0):
    m = mask.numpy()
    h, w = m.shape
    for yy in range(h):
        for xx in range(w):
            if not m[yy, xx]:
                continue
            if xx == 0 or not m[yy, xx - 1]:
                draw.line([(xx * S + off, yy * S), (xx * S + off, (yy + 1) * S)], color, width)
            if xx == w - 1 or not m[yy, xx + 1]:
                draw.line([((xx + 1) * S + off, yy * S), ((xx + 1) * S + off, (yy + 1) * S)], color, width)
            if yy == 0 or not m[yy - 1, xx]:
                draw.line([(xx * S, yy * S + off), ((xx + 1) * S, yy * S + off)], color, width)
            if yy == h - 1 or not m[yy + 1, xx]:
                draw.line([(xx * S, (yy + 1) * S + off), ((xx + 1) * S, (yy + 1) * S + off)], color, width)


try:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    big = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 18)
except OSError:
    font = big = ImageFont.load_default()

P = 32 * S
GAP, HEAD, TXT = 12, 34, 58
cols = ["Target", "Source", "E3 rectangle", "E4 DP edge", "E4-R random (calib.)", "Outlines overlaid"]
W = len(cols) * (P + GAP) + GAP
H = HEAD + len(cases) * (P + TXT + HEAD) + 10
canvas = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(canvas)
for j, c in enumerate(cols):
    d.text((GAP + j * (P + GAP), 8), c, fill="black", font=big)

RED, BLUE, YEL = (230, 30, 30), (20, 90, 255), (255, 210, 0)
for row, (i, label) in enumerate(cases):
    top = HEAD + row * (P + TXT + HEAD)
    d.text((GAP, top + 4), f"Case {label}", fill="black", font=big)
    top += HEAD - 8
    tgt, src = x[i], x[perm[i]]
    panels = [(to_img(tgt), None), (to_img(src), None)]
    stats = ["", ""]
    for meta, col in ((m3, YEL), (m4, RED), (m4r, BLUE)):
        mk = meta["mask"][i, 0]
        mixed = torch.where(mk, src, tgt)
        panels.append((to_img(mixed), (mk, col)))
    def s3():
        return f"D_hat {m3['seam_cost'][i]:.2f}   len {int(m3['seam_count'][i])}\narea {int(m3['mask'][i].sum())}"
    def s4(m):
        st = ["ok", "no patch", "too small", "infeasible", "area fallback"][int(m["dp_status"][i])]
        return (f"D_hat {m['seam_cost'][i]:.2f}   len {int(m['seam_count'][i])}\n"
                f"area {int(m['area_final'][i])}  bends {int(m['bends'][i])}  {st}")
    stats += [s3(), s4(m4), s4(m4r), "yellow E3 / red E4 / blue E4-R"]
    # overlay panel
    ov = to_img(torch.where(m3["mask"][i, 0], src, tgt) * 0.35 + 0.65)
    panels.append((ov, "overlay"))
    for j, (im, extra) in enumerate(panels):
        x0 = GAP + j * (P + GAP)
        im = im.copy()
        dd = ImageDraw.Draw(im)
        if extra == "overlay":
            outline(dd, m3["mask"][i, 0], YEL, 4)
            outline(dd, m4["mask"][i, 0], RED, 3)
            outline(dd, m4r["mask"][i, 0], BLUE, 2)
        elif extra is not None:
            outline(dd, extra[0], extra[1], 3)
        canvas.paste(im, (x0, top))
        d.multiline_text((x0, top + P + 4), stats[j], fill="black", font=font, spacing=3)

canvas.save(OUT)
print("saved", OUT, canvas.size)
for i, label in cases:
    print(label, "| idx", i,
          "| D_hat E3/E4/E4R", [round(float(m["seam_cost"][i]), 2) for m in (m3, m4, m4r)],
          "| bends E4/E4R", int(m4["bends"][i]), int(m4r["bends"][i]),
          "| len", [int(m["seam_count"][i]) for m in (m3, m4, m4r)],
          "| area", int(m3["mask"][i].sum()), int(m4["area_final"][i]), int(m4r["area_final"][i]))
