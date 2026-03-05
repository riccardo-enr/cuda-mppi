#!/usr/bin/env python3
"""Plot MPPI test trajectories from CSV logs."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV_DIR = Path(__file__).parent.parent / "data" / "csv"
PLOT_DIR = Path(__file__).parent.parent / "data" / "plots"


def plot_di_convergence():
    """Plot DoubleIntegrator convergence for MPPI, SMPPI, KMPPI."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("DoubleIntegrator Convergence: MPPI vs SMPPI vs KMPPI", fontsize=14)

    colors = {"MPPI": "#1f77b4", "SMPPI": "#ff7f0e", "KMPPI": "#2ca02c"}

    for name, color in colors.items():
        f = CSV_DIR / f"di_{name}.csv"
        if not f.exists():
            print(f"  Skipping {f} (not found)")
            continue
        d = np.genfromtxt(f, delimiter=",", names=True)

        # XY trajectory
        axes[0, 0].plot(d["px"], d["py"], "-", color=color, label=name, linewidth=1.5)
        axes[0, 0].plot(d["px"][0], d["py"][0], "o", color=color, ms=8)

        # Position norm vs time
        pos_norm = np.sqrt(d["px"] ** 2 + d["py"] ** 2)
        axes[0, 1].plot(d["t"], pos_norm, color=color, label=name)

        # Control ax
        axes[1, 0].plot(d["t"], d["ax"], color=color, label=name, alpha=0.8)

        # Control ay
        axes[1, 1].plot(d["t"], d["ay"], color=color, label=name, alpha=0.8)

    axes[0, 0].plot(0, 0, "r*", ms=15, zorder=5, label="Goal")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[0, 0].set_title("XY Trajectory")
    axes[0, 0].legend()
    axes[0, 0].set_aspect("equal")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("||pos||")
    axes[0, 1].set_title("Position Norm")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("ax [m/s²]")
    axes[1, 0].set_title("Control: ax")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("ay [m/s²]")
    axes[1, 1].set_title("Control: ay")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "di_convergence.png", dpi=150)
    print("  Saved di_convergence.png")


def plot_3d_tracking(csv_name, title, has_ref=True):
    """Plot 3D position tracking + controls (using 2D projections)."""
    f = CSV_DIR / f"{csv_name}.csv"
    if not f.exists():
        print(f"  Skipping {f} (not found)")
        return
    d = np.genfromtxt(f, delimiter=",", names=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(title, fontsize=14)

    # XY trajectory (top-down)
    ax_xy = axes[0, 0]
    ax_xy.plot(d["px"], d["py"], "b-", linewidth=1.5, label="Actual")
    ax_xy.plot(d["px"][0], d["py"][0], "go", ms=10, label="Start")
    ax_xy.plot(d["px"][-1], d["py"][-1], "bs", ms=8, label="End")
    if has_ref:
        ax_xy.plot(d["ref_px"], d["ref_py"], "r--", linewidth=1, alpha=0.6, label="Reference")
        ax_xy.plot(d["ref_px"][-1], d["ref_py"][-1], "r*", ms=15, label="Goal")
    else:
        ax_xy.plot(0, 0, "r*", ms=15, label="Goal (origin)")
    ax_xy.set_xlabel("X [m]")
    ax_xy.set_ylabel("Y [m]")
    ax_xy.set_title("XY Trajectory")
    ax_xy.legend(fontsize=7)
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.3)

    # XZ trajectory (side view)
    ax_xz = axes[0, 1]
    ax_xz.plot(d["px"], d["pz"], "b-", linewidth=1.5, label="Actual")
    ax_xz.plot(d["px"][0], d["pz"][0], "go", ms=10)
    ax_xz.plot(d["px"][-1], d["pz"][-1], "bs", ms=8)
    if has_ref:
        ax_xz.plot(d["ref_px"], d["ref_pz"], "r--", linewidth=1, alpha=0.6, label="Reference")
        ax_xz.plot(d["ref_px"][-1], d["ref_pz"][-1], "r*", ms=15)
    else:
        ax_xz.plot(0, 0, "r*", ms=15)
    ax_xz.set_xlabel("X [m]")
    ax_xz.set_ylabel("Z [m]")
    ax_xz.set_title("XZ Trajectory (side)")
    ax_xz.set_aspect("equal")
    ax_xz.grid(True, alpha=0.3)

    # Position components vs time
    ax_pos = axes[0, 2]
    ax_pos.plot(d["t"], d["px"], label="px")
    ax_pos.plot(d["t"], d["py"], label="py")
    ax_pos.plot(d["t"], d["pz"], label="pz")
    if has_ref:
        ax_pos.plot(d["t"], d["ref_px"], "r--", alpha=0.5, label="ref_px")
        ax_pos.plot(d["t"], d["ref_py"], "--", color="orange", alpha=0.5, label="ref_py")
        ax_pos.plot(d["t"], d["ref_pz"], "g--", alpha=0.5, label="ref_pz")
    else:
        ax_pos.axhline(0, color="r", linestyle="--", alpha=0.3, label="ref (origin)")
    ax_pos.set_xlabel("Time [s]")
    ax_pos.set_ylabel("Position [m]")
    ax_pos.set_title("Position vs Time")
    ax_pos.legend(fontsize=7)
    ax_pos.grid(True, alpha=0.3)

    # Velocity
    ax_vel = axes[1, 0]
    ax_vel.plot(d["t"], d["vx"], label="vx")
    ax_vel.plot(d["t"], d["vy"], label="vy")
    ax_vel.plot(d["t"], d["vz"], label="vz")
    ax_vel.set_xlabel("Time [s]")
    ax_vel.set_ylabel("Velocity [m/s]")
    ax_vel.set_title("Velocity vs Time")
    ax_vel.legend(fontsize=7)
    ax_vel.grid(True, alpha=0.3)

    # Control
    ax_ctrl = axes[1, 1]
    ax_ctrl.plot(d["t"], d["ax"], label="ax")
    ax_ctrl.plot(d["t"], d["ay"], label="ay")
    ax_ctrl.plot(d["t"], d["az"], label="az")
    ax_ctrl.set_xlabel("Time [s]")
    ax_ctrl.set_ylabel("Acceleration [m/s²]")
    ax_ctrl.set_title("Control Inputs")
    ax_ctrl.legend(fontsize=7)
    ax_ctrl.grid(True, alpha=0.3)

    # Position error norm
    ax_err = axes[1, 2]
    if has_ref:
        err = np.sqrt((d["px"] - d["ref_px"])**2 +
                      (d["py"] - d["ref_py"])**2 +
                      (d["pz"] - d["ref_pz"])**2)
    else:
        err = np.sqrt(d["px"]**2 + d["py"]**2 + d["pz"]**2)
    ax_err.plot(d["t"], err, "b-", linewidth=1.5)
    ax_err.set_xlabel("Time [s]")
    ax_err.set_ylabel("||error|| [m]")
    ax_err.set_title("Tracking Error")
    ax_err.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{csv_name}.png", dpi=150)
    print(f"  Saved {csv_name}.png")


def main():
    if not CSV_DIR.exists():
        print(f"Error: {CSV_DIR} does not exist. Run mppi_log_trajectories first.")
        sys.exit(1)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Plotting DoubleIntegrator convergence...")
    plot_di_convergence()

    print("Plotting static point tracking...")
    plot_3d_tracking("static_tracking", "Static Point Tracking — AccelerationTrackingCost")

    print("Plotting waypoint sequence...")
    plot_3d_tracking("waypoint_sequence", "Waypoint Sequence Tracking — A→B")

    print("Plotting hover fallback...")
    plot_3d_tracking("hover_fallback", "Hover at Origin (nullptr fallback)", has_ref=False)

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
