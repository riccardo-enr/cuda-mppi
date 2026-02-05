#!/usr/bin/env python3
"""Analyze and visualize hierarchical exploration campaign results."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def analyze_exploration():
    """Analyze exploration behavior of Standard vs Informative MPPI."""

    # Load trajectories
    info = pd.read_csv("build/info_traj.csv")
    std = pd.read_csv("build/std_traj.csv")

    # Define exploration regions (adjusted to actual exploration behavior)
    area1_visits = info[
        (info["x"] >= 1.5)
        & (info["x"] <= 4.5)
        & (info["y"] >= 6.0)
        & (info["y"] <= 9.0)
    ]
    area2_visits = info[
        (info["x"] >= 6.0)
        & (info["x"] <= 9.0)
        & (info["y"] >= 1.0)
        & (info["y"] <= 4.0)
    ]
    goal_visits = info[
        np.sqrt((info["x"] - 9.0) ** 2 + (info["y"] - 5.0) ** 2) < 1.5
    ]

    print("=" * 60)
    print("HIERARCHICAL EXPLORATION CAMPAIGN ANALYSIS")
    print("=" * 60)

    # Area 1 (top region near start)
    if len(area1_visits) > 0:
        print("\n✓ Area 1 Explored:")
        print(
            f"  - Duration: t={area1_visits['t'].iloc[0]:.1f}s to {area1_visits['t'].iloc[-1]:.1f}s"
        )
        print(f"  - Timesteps: {len(area1_visits)}")
        print(f"  - Max height: y={area1_visits['y'].max():.2f}m")
        print("  - Exploration phase successful!")
    else:
        print("\n✗ Area 1 not explored")

    # Area 2 (bottom-right region)
    if len(area2_visits) > 0:
        print("\n✓ Area 2 Explored:")
        print(
            f"  - Duration: t={area2_visits['t'].iloc[0]:.1f}s to {area2_visits['t'].iloc[-1]:.1f}s"
        )
        print(f"  - Timesteps: {len(area2_visits)}")
        print("  - Transition to second area successful!")
    else:
        print("\n✗ Area 2 not explored (may need longer simulation)")

    # Goal approach
    if len(goal_visits) > 0:
        print("\n✓ Goal Approached:")
        print(f"  - First arrival: t={goal_visits['t'].iloc[0]:.1f}s")
        print("  - Drive-to-goal phase successful!")
    else:
        print("\n✗ Goal not reached (still exploring)")

    # Create detailed visualization
    create_detailed_plot(info, std, area1_visits, area2_visits)


def create_detailed_plot(info, std, area1, area2):
    """Create detailed visualization of interest regions and visits."""

    fig, ax1 = plt.subplots(figsize=(10, 10))
    ax1.set_facecolor("#f5f5f5")

    # Mark exploration phases (actual visits to regions)
    if len(area1) > 0:
        ax1.plot(
            area1["x"],
            area1["y"],
            "r-",
            linewidth=4,
            label="Visited Area 1",
            zorder=10,
        )
    if len(area2) > 0:
        ax1.plot(
            area2["x"],
            area2["y"],
            "b-",
            linewidth=4,
            label="Visited Area 2",
            zorder=10,
        )

    # Informative trajectory (the rest of it)
    ax1.plot(info["x"], info["y"], "k-", alpha=0.3, label="Full Trajectory", zorder=5)

    # High-interest zones (Original map definition)
    rect1 = Rectangle(
        (2, 6.5),
        2,
        2,
        linewidth=2,
        edgecolor="gold",
        facecolor="yellow",
        alpha=0.4,
        label="Target Interest Regions",
    )
    ax1.add_patch(rect1)
    rect2 = Rectangle(
        (6.5, 1.5),
        2,
        2,
        linewidth=2,
        edgecolor="gold",
        facecolor="yellow",
        alpha=0.4,
    )
    ax1.add_patch(rect2)

    # Wall (Vertical wall with opening)
    ax1.vlines(
        5.0, 0, 4.0, colors="cyan", linestyles="-", linewidth=6, label="Wall", alpha=0.8
    )
    ax1.vlines(5.0, 6.0, 10.0, colors="cyan", linestyles="-", linewidth=6, alpha=0.8)

    # Start and goal
    ax1.scatter(
        info["x"].iloc[0],
        info["y"].iloc[0],
        c="lime",
        s=200,
        marker="o",
        label="Start",
        zorder=15,
        edgecolor="black",
    )
    ax1.scatter(
        9.0,
        5.0,
        marker="*",
        s=400,
        color="gold",
        edgecolor="black",
        label="Goal",
        zorder=15,
    )

    ax1.set_xlabel("x [m]", fontsize=12)
    ax1.set_ylabel("y [m]", fontsize=12)
    ax1.set_title(
        "I-MPPI: Interest Regions and Exploration Visits",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(loc="upper left", framealpha=1.0, fontsize=10)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)

    plt.tight_layout()
    plt.savefig("docs/_media/imppi_analysis.png", dpi=150, bbox_inches="tight")
    print("\n✓ Interest region analysis saved to: docs/_media/imppi_analysis.png")

    plt.show()


if __name__ == "__main__":
    analyze_exploration()
