"""Paired CIFAR-100 boundary audit, extending C's diagnostic with tracked padding."""
import argparse
import csv
import hashlib
import json
import sys
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from augmentations import cutmix_batch
from augmentations.seam_dp import STATUS_ACCEPTED, STATUS_AREA, STATUS_NAMES, STATUS_NO_PATCH


ARMS = {"E1": "random", "E2": "dual_edge", "E3": "seam_aware", "E4": "seam_dp", "E4-R": "seam_random"}
SETTINGS = dict(p=1.0, alpha=1.0, k=4, gamma=0.5, beta=1.0, seam_radius=2, seam_mu=0.05,
                seam_min_width=2, seam_area_budget=0.2, seam_random_mu=0.225)


class TrackPadding:
    """Apply the training geometry to RGB and an extra validity channel together."""

    def __init__(self):
        self.augment = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                           transforms.RandomHorizontalFlip()])
        self.tensor = transforms.ToTensor()

    def __call__(self, image):
        rgb = self.tensor(image)
        augmented = self.augment(torch.cat((rgb, torch.ones_like(rgb[:1]))))
        return augmented[:3], ~augmented[3].bool()


def seam_totals(images, mask, padded):
    """Return sums and counts of internal four-neighbor pairs in four populations."""
    sums = images.new_zeros((len(images), 4), dtype=torch.float64)
    counts = torch.zeros_like(sums, dtype=torch.int64)
    for vertical in (False, True):
        rgb = images.transpose(-1, -2) if vertical else images
        mk = mask.transpose(-1, -2) if vertical else mask
        pad = padded.transpose(-1, -2) if vertical else padded
        cross = mk[..., :-1] != mk[..., 1:]
        jump = (rgb[..., :-1] - rgb[..., 1:]).abs().mean(1)
        left_pad, right_pad = pad[..., :-1], pad[..., 1:]
        black = (rgb == 0).all(1)
        for col, selected in enumerate((cross, cross & ~(left_pad & right_pad),
                                       cross & ~(left_pad | right_pad),
                                       cross & ~(black[..., :-1] & black[..., 1:]))):
            sums[:, col] += (jump * selected).flatten(1).sum(1, dtype=torch.float64)
            counts[:, col] += selected.flatten(1).sum(1)
    return sums, counts


def audit_geometry(mask, meta):
    """Verify accepted geometry from actual mask rows and exact rectangle fallback."""
    top, left, bottom, right = meta["boxes"].unbind(1)
    height, width = mask.shape[-2:]
    y = torch.arange(height, device=mask.device)
    x = torch.arange(width, device=mask.device)
    vertical = (top[:, None] <= y) & (y < bottom[:, None])
    start = torch.where(mask, x, width).amin(2)
    end = torch.where(mask, x, -1).amax(2) + 1
    reconstructed = vertical[:, :, None] & (start[:, :, None] <= x) & (x < end[:, :, None])
    base = vertical[:, :, None] & (left[:, None, None] <= x) & (x < right[:, None, None])
    valid_row = ((end - start >= SETTINGS["seam_min_width"])
                 & ((start - left[:, None]).abs() <= SETTINGS["seam_radius"])
                 & ((end - right[:, None]).abs() <= SETTINGS["seam_radius"]))
    adjacent = vertical[:, 1:] & vertical[:, :-1]
    dx_left, dx_right = start.diff(dim=1).abs(), end.diff(dim=1).abs()
    area = mask.flatten(1).sum(1)
    base_area = base.flatten(1).sum(1)
    status = meta["dp_status"]
    valid = ((reconstructed == mask).flatten(1).all(1)
             & (~vertical | valid_row).all(1)
             & (~adjacent | ((dx_left <= 1) & (dx_right <= 1))).all(1)
             & ((area - base_area).abs() <= SETTINGS["seam_area_budget"] * base_area.double() + 1e-9))
    changed = (mask != base).flatten(1).any(1)
    bends = ((dx_left + dx_right) * adjacent).sum(1)
    if (((status == STATUS_ACCEPTED) & ~valid) | ((status != STATUS_ACCEPTED) & changed)).any():
        raise ValueError("Invalid accepted geometry or fallback mask")
    if not (torch.equal(bends, meta["bends"]) and torch.equal(base_area, meta["area_base"])
            and torch.equal(changed, meta["dp_changed"]) and torch.equal(area, meta["area_final"])):
        raise ValueError("Recomputed geometry disagrees with C metadata")
    moved_to_edge = ((mask[:, :, 0].any(1) & (left != 0))
                     | (mask[:, :, -1].any(1) & (right != width)))
    return base_area, changed, moved_to_edge & changed, bends, status


def collect_batch(images, labels, padding, seed):
    results = {}
    for name, policy in ARMS.items():
        np.random.seed(seed)
        torch.manual_seed(seed)
        mixed, label_a, label_b, lam, meta = cutmix_batch(images, labels, policy=policy, **SETTINGS)
        mask = meta["mask"][:, 0]
        perm = meta["permutation"]
        area = mask.flatten(1).sum(1)
        if not (torch.equal(mixed, torch.where(mask[:, None], images[perm], images))
                and torch.equal(label_a, labels) and torch.equal(label_b, labels[perm])
                and torch.allclose(lam, 1 - area.float() / mask[0].numel(), atol=1e-7, rtol=0)):
            raise ValueError(f"{name}: final mask or label mismatch")
        sums, counts = seam_totals(mixed, mask, torch.where(mask, padding[perm], padding))
        if "seam_count" in meta and not torch.equal(counts[:, 0], meta["seam_count"]):
            raise ValueError(f"{name}: raw boundary count differs from C metadata")
        geometry = audit_geometry(mask, meta) if name in ("E4", "E4-R") else None
        results[name] = (mixed, mask, meta, area, sums, counts, geometry)
    ref = results["E1"]
    for name, item in results.items():
        meta = item[2]
        if not torch.equal(meta["permutation"], ref[2]["permutation"]):
            raise ValueError(f"{name}: donor permutation differs")
        if name in ("E4", "E4-R"):
            if not torch.equal(meta["boxes"], results["E3"][2]["boxes"]):
                raise ValueError(f"{name}: base rectangle differs from E3")
        elif not torch.equal(item[3], ref[3]):
            raise ValueError(f"{name}: area differs from E1")
    return results


VARIANTS = ("all", "without_double_padding", "without_any_padding", "without_double_black")


def sample_rows(results, start):
    for name, (_, _, _, area, sums, counts, geometry) in results.items():
        for index in range(len(area)):
            row = {"arm": name, "sample": start + index, "area_ratio": area[index].item() / 1024}
            for col, variant in enumerate(VARIANTS):
                row[f"count_{variant}"] = int(counts[index, col])
                row[f"sum_{variant}"] = float(sums[index, col])
            if geometry:
                base, changed, touched, bends, status = (int(values[index]) for values in geometry)
                row.update(base_area=base, changed=changed, touched_edge=touched,
                           bends=bends, status=STATUS_NAMES[status])
            yield row


def summarize(rows):
    def fraction(numerator, denominator):
        return numerator / denominator if denominator else None

    total = sum(row["count_all"] for row in rows)
    summary = {"samples": len(rows), "area_ratio": float(np.mean([row["area_ratio"] for row in rows])),
               "boundary_length": fraction(total, sum(row["count_all"] > 0 for row in rows))}
    summary["seam"] = {}
    for variant in VARIANTS:
        count_key, sum_key = f"count_{variant}", f"sum_{variant}"
        valid = [row for row in rows if row[count_key]]
        counts = sum(row[count_key] for row in valid)
        summary["seam"][variant] = {
            "pairs": counts, "na_images": len(rows) - len(valid),
            "pair_weighted_rgb": fraction(sum(row[sum_key] for row in valid), counts),
            "image_weighted_rgb": fraction(sum(row[sum_key] / row[count_key] for row in valid), len(valid)),
            "removed_pair_fraction": fraction(total - counts, total),
        }
    if "status" in rows[0]:
        attempted = [row for row in rows if row["status"] != "no_patch"]
        summary.update(
            status_counts={status: sum(row["status"] == status for row in rows) for status in STATUS_NAMES},
            fallback_rate=fraction(sum(row["status"] != "accepted" for row in attempted), len(attempted)),
            bends=fraction(sum(row["bends"] for row in attempted), len(attempted)),
            final_over_base_area=fraction(sum(row["area_ratio"] * 1024 / row["base_area"] for row in attempted),
                                          len(attempted)),
            edge_snap_rate_among_changed=fraction(sum(row["touched_edge"] for row in rows),
                                                  sum(row["changed"] for row in rows)),
        )
    return summary


def update_examples(examples, batch, results, start):
    images = batch
    e4, e4r, e3 = results["E4"], results["E4-R"], results["E3"]
    pair_count = e4[5][:, 0].clamp_min(1)
    gain = e3[4][:, 0] / e3[5][:, 0].clamp_min(1) - e4[4][:, 0] / pair_count
    status = e4[-1][-1]
    snap = e4[-1][2]
    padding_fraction = (e4[5][:, 0] - e4[5][:, 2]) / pair_count
    for key, metric, valid in (
        ("DP lowers seam", gain, (status == STATUS_ACCEPTED) & (e4[5][:, 0] > 0) & (gain > 0)),
        ("Area-budget fallback", e4[-1][0].float(), status == STATUS_AREA),
        ("Boundary pushed to image edge", gain, snap),
        ("Seam touches Crop padding", padding_fraction, (padding_fraction > 0) & (e4r[5][:, 0] > 0)),
    ):
        if not valid.any():
            continue
        i = int(metric.masked_fill(~valid, -float("inf")).argmax())
        if key not in examples or float(metric[i]) > examples[key][0]:
            donor = int(e3[2]["permutation"][i])
            panels = [images[i], images[donor], *(result[0][i] for result in results.values())]
            masks = [result[1][i] for result in results.values()]
            examples[key] = (float(metric[i]), start + i, torch.stack(panels), masks)


def save_example_grid(path, examples):
    from PIL import ImageDraw, ImageFont

    scale = 4
    side = 32 * scale
    names = ["Target", "Source", *ARMS]
    width, height = len(names) * (side + 8) + 12, 26 + len(examples) * (side + 32)
    canvas = Image.new("RGB", (width, height), "white")
    drawing = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for col, name in enumerate(names):
        drawing.text((8 + col * (side + 8), 5), name, fill="#252a32", font=font)
    for row, (case, (_, _, panels, masks)) in enumerate(examples.items()):
        y = 24 + row * (side + 32)
        drawing.text((8, y), case, fill="#252a32", font=font)
        for col, tensor in enumerate(panels):
            im = Image.fromarray((tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8))
            panel = im.resize((side, side), Image.Resampling.NEAREST)
            if col >= 2:
                mask = masks[col - 2]
                lines = ImageDraw.Draw(panel)
                color = "#58f0de" if col < 5 else "#ffd257"
                for yy, xx in torch.nonzero(mask[:, 1:] != mask[:, :-1]).tolist():
                    px = (xx + 1) * scale
                    lines.line((px, yy * scale, px, (yy + 1) * scale), fill=color, width=2)
                for yy, xx in torch.nonzero(mask[1:, :] != mask[:-1, :]).tolist():
                    py = (yy + 1) * scale
                    lines.line((xx * scale, py, (xx + 1) * scale, py), fill=color, width=2)
            canvas.paste(panel, (8 + col * (side + 8), y + 18))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/member_d")
    parser.add_argument("--example-figure", type=Path, default=ROOT / "results/figures/d_mixed_examples.png")
    parser.add_argument("--batches", type=int, default=20)
    args = parser.parse_args()
    if args.batches < 1:
        parser.error("--batches must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    split = ROOT / "data/splits/train_indices.npy"
    dataset = datasets.CIFAR100(str(args.data_root), train=True, download=False, transform=TrackPadding())
    loader = DataLoader(Subset(dataset, np.load(split)), batch_size=128, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(42))
    rows, examples, start = [], {}, 0
    batches = list(islice(loader, args.batches))
    with torch.inference_mode():
        for batch_number, ((images, padding), labels) in enumerate(batches):
            results = collect_batch(images, labels, padding, 1000 + batch_number)
            rows.extend(sample_rows(results, start))
            update_examples(examples, images, results, start)
            start += len(images)
    mask_fields = ["base_area", "changed", "touched_edge", "bends", "status"]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "boundary_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*rows[0], *mask_fields], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = {"dataset": "CIFAR-100 fixed training split", "split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
              "seed": 42, "batch_size": 128, "batches": batch_number + 1, "device": "cpu",
              "torch": str(torch.__version__), "settings": SETTINGS,
              "examples": {key: {"score": value[0], "sample": value[1]} for key, value in examples.items()},
              "arms": {name: summarize([row for row in rows if row["arm"] == name]) for name in ARMS}}
    (args.output / "boundary_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save_example_grid(args.example_figure, examples)
    print(f"Audited {start:,} inputs per arm")


if __name__ == "__main__":
    main()
