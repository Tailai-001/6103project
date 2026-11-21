"""Compile the 18 formal A/B/C validation runs and derive report-ready evidence."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
B = RESULTS / "member_b"
C = RESULTS / "member_c"
A = ROOT / "Baseline Implementation & Training Infrastructure (Member A)"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def split_hashes():
    hashes = {name: hashlib.sha256((ROOT / "data/splits" / name).read_bytes()).hexdigest()
              for name in ("train_indices.npy", "val_indices.npy")}
    for name, digest in hashes.items():
        if hashlib.sha256((A / "data/splits" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Member A and shared split disagree for {name}")
    return hashes


def formal_log(path):
    rows = read_csv(path)
    if ([int(r["epoch"]) for r in rows] != list(range(1, 201)) or
            any(int(r["train_samples"]) != 45000 for r in rows)):
        raise ValueError(f"{path}: incomplete formal run")
    return rows


def timing(path):
    return {r["config"]: r["augmentation_ms_per_batch"] for r in read_json(path)["results"]}


def build_runs():
    rows = []
    split = split_hashes()
    registry = {r["run_id"]: r for r in read_csv(ROOT / "experiments/registry.csv")}
    a_dep = (A / "requirements-lock.txt").read_text()
    a_torch = next(line.split("==", 1)[1] for line in a_dep.splitlines() if line.startswith("torch=="))

    def add(member, machine, experiment, role, run_id, policy, path, aug,
            published=None, checkpoint=None, env=None, aug_ms=None):
        logs = formal_log(path)
        best = max(logs, key=lambda r: float(r["val_accuracy"]))
        val = 100 * float(best["val_accuracy"])
        last10 = 100 * mean(float(r["val_accuracy"]) for r in logs[-10:])
        f1, f1_device, reeval_val, note = None, None, None, ""
        if checkpoint:
            reeval_val = checkpoint["val_top1_percent"]
            if checkpoint["samples"] != 5000 or checkpoint["checkpoint_epoch"] != int(best["epoch"]):
                raise ValueError(f"{run_id}: checkpoint provenance mismatch")
            if abs(reeval_val - val) > 1e-8:
                if not (run_id == "c_e4r_k4_s42" and abs(reeval_val - val) < 0.02000001
                        and checkpoint.get("device") == "cpu"):
                    raise ValueError(f"{run_id}: unexpected checkpoint accuracy mismatch")
                note = "CPU re-evaluation Top-1 differs by one image; best Top-1 is from the original GPU log."
            f1 = checkpoint["val_macro_f1"]
            f1_device = checkpoint.get("device", "cuda (B re-evaluation)")
        if published:
            if abs(float(published["best_val_top1_percent"]) - val) > 1e-8 or int(published["best_epoch"]) != int(best["epoch"]):
                raise ValueError(f"{run_id}: raw log and summary disagree")
            if "last10_mean_val_top1_percent" in published and abs(float(published["last10_mean_val_top1_percent"]) - last10) > 1e-8:
                raise ValueError(f"{run_id}: raw log and last-10 mean disagree")
            if env["split_sha256"] != split:
                raise ValueError(f"{run_id}: split hashes disagree")
            registered = registry[run_id]
            if registered["status"] != "completed" or "phase=formal" not in registered["notes"]:
                raise ValueError(f"{run_id}: expected completed formal run")
            if abs(float(registered["augmentation_ms_per_batch"]) - float(aug_ms)) > 1e-8:
                raise ValueError(f"{run_id}: registry and timing evidence disagree")
        else:
            registered = {}
        effective_k = aug.get("k") if policy not in ("none", "random") else None
        uses_seam = policy in ("seam_aware", "seam_dp", "seam_random")
        uses_boundary = policy in ("seam_dp", "seam_random")
        rows.append((dict(
            experiment=experiment, role=role, run_id=run_id, source=member, policy=policy,
            k=effective_k, gamma=aug.get("gamma") if effective_k else None,
            beta=aug.get("beta") if uses_seam or (machine == "C-A2" and experiment == "E2") else None,
            radius=aug.get("seam_radius") if uses_boundary else None,
            mu=aug.get("seam_mu") if uses_boundary else None, random_mu=aug.get("seam_random_mu"),
            min_width=aug.get("seam_min_width") if uses_boundary else None,
            area_budget=aug.get("seam_area_budget") if uses_boundary else None,
            alpha=aug.get("alpha") if policy != "none" else None, p=aug.get("p") if policy != "none" else None,
            seed=int(published["seed"]) if published else 42, epochs=len(logs), train_samples=45000,
            val_samples=5000, split="CIFAR-100 fixed stratified 45k/5k",
            train_indices_sha256=split["train_indices.npy"], val_indices_sha256=split["val_indices.npy"],
            machine=machine, gpu=env["gpu"], torch=env["torch"], python=env.get("python"),
            best_val_top1_percent=round(val, 3), best_epoch=int(best["epoch"]),
            last10_val_top1_mean_percent=round(last10, 5), val_macro_f1=f1,
            macro_f1_eval_device=f1_device, checkpoint_reeval_val_top1_percent=reeval_val,
            sum_epoch_seconds=round(sum(float(r["epoch_seconds"]) for r in logs), 3),
            full_process_wall_seconds=float(registered["total_training_seconds"]) if published else None,
            augmentation_ms_per_batch=aug_ms, test_top1_percent=None, test_macro_f1=None,
            log_path=path.relative_to(ROOT).as_posix(),
            training_git_commit=registered.get("git_commit") or None,
            config_path=registered.get("config_path", (A / "configs/main.yaml").relative_to(ROOT).as_posix()),
            notes=note,
        ), logs))

    for experiment, policy in (("E0", "none"), ("E1", "random")):
        run_id = experiment.lower() + "_final_s42"
        path = A / "results/baselines" / f"{run_id}.csv"
        add("A", "A-original", experiment, "baseline", run_id, policy, path, dict(alpha=1, p=0.5),
            env={"gpu": "NVIDIA A40", "torch": a_torch})

    b_eval = {r["run_id"]: r for r in read_json(B / "checkpoint_validation.json")}
    for published in read_csv(TABLES / "member_b_validation_results.csv"):
        run_id = published["run_id"]
        meta = B / "run_metadata" / run_id
        aug = read_json(meta / "config.json")["augmentation"]
        experiment = "E1-equivalent" if int(published["k"]) == 1 else registry[run_id]["experiment"]
        role = "K=1 control" if experiment == "E1-equivalent" else "main" if aug["k"] == 4 else "K ablation"
        add("B", "B-original", experiment, role, run_id, published["policy"],
            B / "raw_logs" / Path(published["log_path"]).name, aug, published, b_eval[run_id],
            read_json(meta / "environment.json"), float(published["augmentation_ms_per_batch"]))

    c_eval = {r["run_id"]: r for r in read_json(C / "checkpoint_validation.json")}
    c_timing = {machine: timing(path) for machine, path in (
        ("C-main", C / "aug_timing.json"), ("C-A2", C / "machine_A2/aug_timing.json"),
        ("C-B", C / "machine_B/aug_timing.json"),
    )}
    for published in read_csv(C / "summary.csv"):
        run_id = published["run_id"]
        meta = C / "run_metadata" / run_id
        aug = read_json(meta / "config.json")["augmentation"]
        experiment = published["experiment"].split()[0]
        if run_id in ("c_e3_k4_s42", "c_e4_k4_s42", "c_e4r_k4_s42"):
            machine, role = "C-main", "main"
        elif "ctrl_" in run_id:
            machine, role = "C-A2", "same GPU model control"
        elif run_id.startswith("c_e3_"):
            machine, role = "C-A2", "beta ablation"
        else:
            machine, role = "C-B", "r/mu ablation"
        add("C", machine, experiment, role, run_id, published["policy"],
            C / "raw_logs" / Path(published["log_path"]).name, aug, published, c_eval[run_id],
            read_json(meta / "environment.json"), c_timing[machine][Path(published["config"]).stem])
    return rows


def format_value(value, decimals=2):
    return "N/A" if value is None else f"{value:.{decimals}f}"


def write_tables(runs):
    rows = [r for r, _ in runs]
    write_csv(TABLES / "main_results.csv", rows)
    text = [
        "# Main results", "",
        "CIFAR-100, fixed 45,000/5,000 train/validation split, seed 42, 200 epochs. Top-1 values are validation percentages.",
        "Machine labels identify distinct hosts or original environments. A's original run times are sums of epoch durations;",
        "B/C times are complete process wall times. Timing values should be compared within their measurement host.", "",
        "| Experiment | Run | Machine / GPU | Best Val Top-1 (%) | Last 10 Mean (%) | Val Macro-F1 | Time (min) | Aug. (ms/batch) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        f1 = format_value(row["val_macro_f1"], 4)
        if row["macro_f1_eval_device"] == "cpu":
            f1 += " (CPU)"
        seconds = row["full_process_wall_seconds"] or row["sum_epoch_seconds"]
        text.append(f'| {row["experiment"]} | `{row["run_id"]}` | {row["machine"]} / {row["gpu"]} | '
                    f'{row["best_val_top1_percent"]:.2f} | {row["last10_val_top1_mean_percent"]:.2f} | {f1} | '
                    f'{seconds / 60:.2f} | {format_value(row["augmentation_ms_per_batch"])} |')
    text += ["", "C-main E4-R was 79.46% in the training log and 79.44% in CPU checkpoint re-evaluation (one image).",
             "The table uses the training-log Top-1. Official test scores have not been measured in this handoff."]
    (TABLES / "main_results.md").write_text("\n".join(text) + "\n", encoding="utf-8")

    designs = [
        ("K (edge-guided)", lambda r: r["source"] == "B"),
        ("β (rectangle seam)", lambda r: r["experiment"] in ("E2", "E3") and r["source"] == "C"),
        ("r / μ (DP boundary)", lambda r: r["experiment"] == "E4"),
    ]
    ablations = [dict(factor=label, **{key: row[key] for key in (
        "run_id", "experiment", "machine", "k", "beta", "radius", "mu",
        "best_val_top1_percent", "last10_val_top1_mean_percent", "val_macro_f1",
        "augmentation_ms_per_batch", "log_path",
    )}) for label, include in designs for row in rows if include(row)]
    write_csv(TABLES / "d_ablation.csv", ablations)
    out = ["# Ablations", "",
           "Seed 42, fixed validation split. Keep machine labels when comparing rows. C-A2 E2 is the β=0 reference.",
           "The r=3 row was selected after viewing these validation runs.", "",
           "| Factor | Run | Machine | K | β | r | μ | Best Val Top-1 (%) | Last 10 Mean (%) | Val Macro-F1 |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in ablations:
        settings = " | ".join("—" if row[k] is None else str(row[k]) for k in ("k", "beta", "radius", "mu"))
        out.append(f'| {row["factor"]} | `{row["run_id"]}` | {row["machine"]} | {settings} | '
                   f'{row["best_val_top1_percent"]:.2f} | {row["last10_val_top1_mean_percent"]:.2f} | {row["val_macro_f1"]:.4f} |')
    (TABLES / "d_ablation.md").write_text("\n".join(out) + "\n", encoding="utf-8")


def plot_report(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    figure, panels = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(11.5, 7), layout="constrained")
    for panel, host, title in zip(panels.flat, ("A-original", "B-original", "C-main", "C-A2"),
                                  ("A · A40", "B · A800", "C-main · RTX 4090", "C-A2 · separate RTX 4090")):
        for r, logs in runs:
            if r["machine"] != host or r["role"] in ("K ablation", "beta ablation"):
                continue
            label = r["experiment"]
            panel.plot([int(x["epoch"]) for x in logs], [100 * float(x["val_accuracy"]) for x in logs],
                       linewidth=1.35, label=label)
        panel.set_title(title, loc="left")
        panel.grid(alpha=0.23)
        panel.legend(loc="lower right", fontsize=8)
    figure.supxlabel("Epoch")
    figure.supylabel("Clean validation Top-1 (%)")
    figure.suptitle("CIFAR-100 · fixed validation split · seed 42")
    figure.savefig(FIGURES / "d_validation_curves.png", dpi=170)
    plt.close(figure)

    times = timing(C / "aug_timing.json")
    keys = ("b_dual_edge_k4_s42", "c_seam_aware_k4_s42", "c_seam_dp_k4_s42", "c_seam_random_k4_s42")
    timing_rows = [dict(method=method, config=key, machine="C-main / RTX 4090",
                        augmentation_ms_per_batch=times[key], source="results/member_c/aug_timing.json")
                   for method, key in zip(("E2", "E3", "E4", "E4-R"), keys)]
    a2_times = timing(C / "machine_A2/aug_timing.json")
    a2_keys = ("c_ctrl_random_s42", "c_ctrl_dual_edge_k4_s42", *keys[1:])
    a2_rows = [dict(method=method, config=key, machine="C-A2 / separate RTX 4090",
                    augmentation_ms_per_batch=a2_times[key],
                    source="results/member_c/machine_A2/aug_timing.json")
               for method, key in zip(("E1", "E2", "E3", "E4", "E4-R"), a2_keys)]
    write_csv(TABLES / "d_efficiency.csv", timing_rows + a2_rows)
    rows = {r["experiment"]: r for r, _ in runs if r["machine"] == "C-main"}
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    values = [r["augmentation_ms_per_batch"] for r in timing_rows]
    bars = left.bar([r["method"] for r in timing_rows], values, color="#296b8a")
    left.bar_label(bars, fmt="%.2f", padding=3)
    left.set(ylabel="Mean ms per batch", ylim=(0, max(values) * 1.18), title="Synchronized augmentation")
    times_min = [rows[method]["full_process_wall_seconds"] / 60 for method in ("E3", "E4", "E4-R")]
    bars = right.bar(("E3", "E4", "E4-R"), times_min, color="#56785e")
    right.bar_label(bars, fmt="%.2f", padding=3)
    right.set(ylabel="Minutes", ylim=(0, max(times_min) * 1.18), title="Full 200-epoch process")
    for ax in (left, right):
        ax.grid(axis="y", alpha=0.2)
    figure.suptitle("C-main · same RTX 4090 host")
    figure.savefig(FIGURES / "d_efficiency.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    runs = build_runs()
    write_tables(runs)
    plot_report(runs)
    print(f"Compiled {len(runs)} formal validation runs")
