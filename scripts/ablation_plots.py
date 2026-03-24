#!/usr/bin/env python3
"""
Ablation comparison plots for I-MPPI study.

Reads CSV outputs from ablation_study.py and generates publication-quality
matplotlib figures.

Usage:
    python ablation_plots.py --input results/ablation/ --output results/ablation/figures/
    python ablation_plots.py --input results/ablation/ --group A
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Color palette (tab10 extended)
COLORS = plt.cm.tab10.colors


def load_runs(input_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all per-run CSVs into a dict keyed by variant name."""
    runs = {}
    for csv_path in sorted(input_dir.glob("*.csv")):
        if csv_path.name == "summary.csv":
            continue
        # Parse: {variant}_r{repeat}_{env}.csv
        stem = csv_path.stem
        parts = stem.rsplit("_", 2)
        if len(parts) < 3:
            continue
        variant = "_".join(parts[:-2])
        df = pd.read_csv(csv_path)
        if variant not in runs:
            runs[variant] = []
        runs[variant].append(df)
    return runs


def plot_coverage_vs_time(
    runs: dict[str, list[pd.DataFrame]],
    output_dir: Path,
    title: str = "Coverage vs Time",
) -> None:
    """Overlay coverage curves for all variants."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (variant, dfs) in enumerate(sorted(runs.items())):
        color = COLORS[i % len(COLORS)]
        for j, df in enumerate(dfs):
            label = variant if j == 0 else None
            alpha = 0.3 if len(dfs) > 1 else 1.0
            ax.plot(
                df["time_s"], df["coverage_pct"] * 100,
                color=color, alpha=alpha, label=label,
            )
        # Mean line if multiple repeats
        if len(dfs) > 1:
            min_len = min(len(df) for df in dfs)
            mean_cov = np.mean(
                [df["coverage_pct"].values[:min_len] for df in dfs], axis=0,
            )
            time_s = dfs[0]["time_s"].values[:min_len]
            ax.plot(time_s, mean_cov * 100, color=color, linewidth=2, label=variant)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Coverage (%)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_vs_time.pdf", dpi=150)
    fig.savefig(output_dir / "coverage_vs_time.png", dpi=150)
    plt.close(fig)
    print("  Saved coverage_vs_time.pdf")


def plot_entropy_vs_time(
    runs: dict[str, list[pd.DataFrame]],
    output_dir: Path,
) -> None:
    """Overlay map entropy curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (variant, dfs) in enumerate(sorted(runs.items())):
        color = COLORS[i % len(COLORS)]
        for j, df in enumerate(dfs):
            label = variant if j == 0 else None
            ax.plot(df["time_s"], df["map_entropy"], color=color, alpha=0.7, label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Map Entropy")
    ax.set_title("Map Entropy vs Time")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "entropy_vs_time.pdf", dpi=150)
    fig.savefig(output_dir / "entropy_vs_time.png", dpi=150)
    plt.close(fig)
    print("  Saved entropy_vs_time.pdf")


def plot_final_coverage_bar(summary: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of final coverage per variant."""
    fig, ax = plt.subplots(figsize=(12, 5))

    variants = summary["variant"]
    coverage = summary["final_coverage"].astype(float) * 100

    colors = [COLORS[i % len(COLORS)] for i in range(len(variants))]
    bars = ax.bar(range(len(variants)), coverage, color=colors)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Final Coverage (%)")
    ax.set_title("Final Coverage by Variant")
    ax.grid(True, alpha=0.3, axis="y")

    # Value labels
    for bar, val in zip(bars, coverage):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=7,
        )

    fig.tight_layout()
    fig.savefig(output_dir / "final_coverage_bar.pdf", dpi=150)
    fig.savefig(output_dir / "final_coverage_bar.png", dpi=150)
    plt.close(fig)
    print("  Saved final_coverage_bar.pdf")


def plot_compute_time_box(
    runs: dict[str, list[pd.DataFrame]],
    output_dir: Path,
) -> None:
    """Box plot of MPPI computation time per variant."""
    fig, ax = plt.subplots(figsize=(12, 5))

    data = []
    labels = []
    for variant in sorted(runs.keys()):
        times = np.concatenate([df["compute_time_ms"].values for df in runs[variant]])
        data.append(times)
        labels.append(variant)

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Computation Time (ms)")
    ax.set_title("MPPI Iteration Time by Variant")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "compute_time_box.pdf", dpi=150)
    fig.savefig(output_dir / "compute_time_box.png", dpi=150)
    plt.close(fig)
    print("  Saved compute_time_box.pdf")


def plot_trajectories(
    runs: dict[str, list[pd.DataFrame]],
    env_name: str,
    output_dir: Path,
) -> None:
    """Top-down trajectory overlay on environment grid."""
    from environments import make_corridor, make_warehouse

    if env_name == "corridor":
        gt_flat, info = make_corridor()
    elif env_name == "warehouse":
        gt_flat, info = make_warehouse()
    else:
        return

    w, h = info["width"], info["height"]
    gt_2d = gt_flat.reshape(h, w)
    ext_x, ext_y = info["extent_m"]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(
        gt_2d, origin="lower", cmap="Greys", alpha=0.4,
        extent=[0, ext_x, 0, ext_y],
    )

    for i, (variant, dfs) in enumerate(sorted(runs.items())):
        color = COLORS[i % len(COLORS)]
        for df in dfs:
            ax.plot(df["pos_x"], df["pos_y"], color=color, linewidth=1.2, alpha=0.8, label=variant)

    # Start marker
    ax.plot(2.0, 2.0, "go", markersize=10, zorder=5, label="Start")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Trajectories — {env_name}")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "trajectories.pdf", dpi=150)
    fig.savefig(output_dir / "trajectories.png", dpi=150)
    plt.close(fig)
    print("  Saved trajectories.pdf")


def main():
    parser = argparse.ArgumentParser(description="I-MPPI Ablation Plots")
    parser.add_argument("--input", type=str, default="results/ablation")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--group", type=str, default=None, help="Filter by group prefix (A, B, C)")
    parser.add_argument("--env", type=str, default="corridor")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(input_dir)
    if args.group:
        runs = {k: v for k, v in runs.items() if k.startswith(args.group)}

    if not runs:
        print(f"No CSV data found in {input_dir}")
        return

    print(f"Loaded {len(runs)} variants from {input_dir}")

    # Summary
    summary_path = input_dir / "summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else None
    if summary is not None and args.group:
        summary = summary[summary["variant"].str.startswith(args.group)]

    plot_coverage_vs_time(runs, output_dir)
    plot_entropy_vs_time(runs, output_dir)
    if summary is not None:
        plot_final_coverage_bar(summary, output_dir)
    plot_compute_time_box(runs, output_dir)
    plot_trajectories(runs, args.env, output_dir)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
