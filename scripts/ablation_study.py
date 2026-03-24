#!/usr/bin/env python3
"""
I-MPPI Ablation Study

Runs the CUDA I-MPPI controller via nanobind with a synthetic
environment, toggling cost layers to isolate each module's contribution.

Usage:
    python ablation_study.py --variants A0 A1 A2 --steps 500 --env corridor
    python ablation_study.py --all --steps 1000 --env warehouse --repeats 3
"""

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from cuda_mppi import (
    InfoField,
    InformativeCost,
    MPPIConfig,
    OccupancyGrid2D,
    QuadrotorDynamics,
    QuadrotorIMPPI,
)
from environments import make_corridor, make_warehouse, simulate_sensor

# ── Ablation variant definitions ──────────────────────────────────────

COST_DEFAULTS = {
    "lambda_local": 10.0,
    "lambda_info": 5.0,
    "target_weight": 1.0,
    "goal_weight": 0.5,
    "collision_penalty": 1000.0,
    "height_weight": 10.0,
    "action_reg": 0.01,
}

MPPI_DEFAULTS = {
    "num_samples": 900,
    "horizon": 50,
    "lambda": 0.5,
    "dt": 0.02,
    "alpha": 0.3,
}

# Group A: cost layer ablation
# Group B: module ablation
# Group C: hyperparameter sensitivity
VARIANTS: dict[str, dict] = {
    # ── Group A: cost layer ablation ──
    "A0_full": {},
    "A1_no_mi": {"lambda_local": 0.0, "lambda_info": 0.0},
    "A2_local_only": {"lambda_info": 0.0},
    "A3_field_only": {"lambda_local": 0.0},
    "A4_no_goal": {"goal_weight": 0.0},
    "A5_no_ref": {"target_weight": 0.0},
    "A6_no_action_reg": {"action_reg": 0.0},
    # ── Group B: module ablation ──
    "B0_with_info_field": {},
    "B1_static_goal": {"use_info_field": False},
    "B2_pure_tracking": {
        "lambda_local": 0.0,
        "lambda_info": 0.0,
        "goal_weight": 0.0,
        "use_info_field": False,
    },
    # ── Group C: hyperparameter sensitivity ──
    "C1_k256": {"num_samples": 256},
    "C2_k512": {"num_samples": 512},
    "C3_k1500": {"num_samples": 1500},
    "C4_h25": {"horizon": 25},
    "C5_h75": {"horizon": 75},
    "C6_lam01": {"lambda": 0.1},
    "C7_lam2": {"lambda": 2.0},
}


# ── Metrics ───────────────────────────────────────────────────────────


@dataclass
class StepMetrics:
    step: int = 0
    time_s: float = 0.0
    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 0.0
    coverage_pct: float = 0.0
    map_entropy: float = 0.0
    frontier_count: int = 0
    path_length: float = 0.0
    compute_time_ms: float = 0.0
    control_smoothness: float = 0.0


@dataclass
class RunMetrics:
    variant: str = ""
    env: str = ""
    repeat: int = 0
    steps: list[StepMetrics] = field(default_factory=list)


def compute_coverage(belief_2d: np.ndarray) -> float:
    """Fraction of cells that are known (|p - 0.5| > 0.3)."""
    return float(np.mean(np.abs(belief_2d - 0.5) > 0.3))


def compute_entropy(belief_2d: np.ndarray) -> float:
    """Shannon entropy of occupancy probabilities."""
    p = belief_2d.flatten()
    # Only non-trivial cells
    mask = (p > 0.01) & (p < 0.99)
    p_m = p[mask]
    if len(p_m) == 0:
        return 0.0
    q_m = 1.0 - p_m
    return float(-np.sum(p_m * np.log(p_m) + q_m * np.log(q_m)))


def compute_frontiers(belief_2d: np.ndarray) -> int:
    """Count free cells adjacent to unknown cells."""
    free = belief_2d < 0.3
    unknown = np.abs(belief_2d - 0.5) < 0.1

    # Pad to handle borders
    padded = np.pad(unknown, 1, constant_values=False)
    has_unknown_neighbor = (
        padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
    )
    return int(np.sum(free & has_unknown_neighbor))


# ── Simulation ────────────────────────────────────────────────────────


def make_mppi_config(overrides: dict) -> MPPIConfig:
    """Create MPPIConfig with defaults + overrides."""
    cfg = MPPIConfig()
    params = {**MPPI_DEFAULTS, **{k: v for k, v in overrides.items() if k in MPPI_DEFAULTS}}
    cfg.num_samples = params["num_samples"]
    cfg.horizon = params["horizon"]
    cfg.lambda_ = params["lambda"]
    cfg.dt = params["dt"]
    cfg.alpha = params["alpha"]
    cfg.nx = 13
    cfg.nu = 4
    cfg.u_scale = 1.0
    return cfg


def make_cost(overrides: dict, env_info: dict) -> InformativeCost:
    """Create InformativeCost with defaults + overrides."""
    cost = InformativeCost()
    params = {**COST_DEFAULTS, **{k: v for k, v in overrides.items() if k in COST_DEFAULTS}}
    cost.lambda_local = params["lambda_local"]
    cost.lambda_info = params["lambda_info"]
    cost.target_weight = params["target_weight"]
    cost.goal_weight = params["goal_weight"]
    cost.collision_penalty = params["collision_penalty"]
    cost.height_weight = params["height_weight"]
    cost.action_reg = params["action_reg"]
    cost.target_altitude = -5.0  # NED

    # Workspace bounds from environment
    ext_x, ext_y = env_info["extent_m"]
    cost.bound_x_min = 0.5
    cost.bound_x_max = ext_x - 0.5
    cost.bound_y_min = 0.5
    cost.bound_y_max = ext_y - 0.5

    # Default goal at far corner
    cost.set_goal(0, ext_x - 2.0, ext_y - 2.0, -5.0)
    cost.num_goals = 1

    return cost


def run_variant(
    variant_name: str,
    overrides: dict,
    env_name: str,
    max_steps: int,
    repeat_idx: int,
) -> RunMetrics:
    """Run a single ablation variant and collect metrics."""
    # Environment
    if env_name == "corridor":
        gt_flat, env_info = make_corridor()
    elif env_name == "warehouse":
        gt_flat, env_info = make_warehouse()
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    w, h = env_info["width"], env_info["height"]
    res = env_info["resolution"]
    gt_2d = gt_flat.reshape(h, w)
    belief_2d = np.full((h, w), 0.5, dtype=np.float32)

    use_info_field = overrides.get("use_info_field", True)

    # MPPI setup
    mppi_cfg = make_mppi_config(overrides)
    dyn = QuadrotorDynamics()
    dyn.mass = 2.0
    dyn.gravity = 9.81
    dyn.tau_omega = 0.05
    cost = make_cost(overrides, env_info)

    mppi = QuadrotorIMPPI(mppi_cfg, dyn, cost)

    # GPU grid for the controller
    grid = OccupancyGrid2D(w, h, res, env_info["origin_x"], env_info["origin_y"])
    info_field = InfoField()

    # Initial state: hover at (2, 2, -5) NED, identity quaternion
    state = np.zeros(13, dtype=np.float32)
    state[0] = 2.0   # px
    state[1] = 2.0   # py
    state[2] = -5.0   # pz (NED, up is negative)
    state[6] = 1.0    # qw

    dt = mppi_cfg.dt
    info_update_interval = 25  # Update info field every 25 steps (~0.5 s)

    metrics = RunMetrics(variant=variant_name, env=env_name, repeat=repeat_idx)
    prev_action = np.zeros(4, dtype=np.float32)
    path_length = 0.0
    prev_pos = state[:3].copy()

    print(f"  [{variant_name}] repeat={repeat_idx}, env={env_name}, steps={max_steps}")

    for step in range(max_steps):
        # 1. Sensor update
        simulate_sensor(gt_2d, belief_2d, state[0], state[1], env_info)

        # 2. Upload belief to GPU
        grid.upload(belief_2d.flatten())
        mppi.update_cost_grid(grid)

        # 3. Info field update (periodic)
        if use_info_field and step % info_update_interval == 0:
            info_field.compute(
                grid, state[0], state[1],
                field_res=0.5, field_extent=5.0,
                n_yaw=8, num_beams=12, max_range=5.0, ray_step=0.1,
            )
            mppi.update_cost_info_field(info_field)

        # 4. MPPI compute
        t0 = time.perf_counter()
        mppi.compute(state)
        compute_ms = (time.perf_counter() - t0) * 1000.0

        action = mppi.get_action()
        mppi.shift()

        # 5. Propagate state
        state = dyn.step(state, action, dt)

        # 6. Metrics
        pos = state[:3]
        path_length += float(np.linalg.norm(pos - prev_pos))
        prev_pos = pos.copy()

        ctrl_smooth = float(np.sum((action - prev_action) ** 2))
        prev_action = action.copy()

        sm = StepMetrics(
            step=step,
            time_s=step * dt,
            pos_x=float(pos[0]),
            pos_y=float(pos[1]),
            pos_z=float(pos[2]),
            coverage_pct=compute_coverage(belief_2d),
            map_entropy=compute_entropy(belief_2d),
            frontier_count=compute_frontiers(belief_2d),
            path_length=path_length,
            compute_time_ms=compute_ms,
            control_smoothness=ctrl_smooth,
        )
        metrics.steps.append(sm)

        # Print progress every 100 steps
        if step % 100 == 0:
            print(
                f"    step {step:4d} | pos=({pos[0]:6.2f}, {pos[1]:6.2f}) | "
                f"cov={sm.coverage_pct:.1%} | entropy={sm.map_entropy:.1f} | "
                f"dt={compute_ms:.1f}ms"
            )

    return metrics


def save_metrics(all_metrics: list[RunMetrics], output_dir: Path) -> None:
    """Save metrics to CSV files, one per variant-repeat."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for run in all_metrics:
        fname = f"{run.variant}_r{run.repeat}_{run.env}.csv"
        fpath = output_dir / fname
        with open(fpath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "step", "time_s", "pos_x", "pos_y", "pos_z",
                "coverage_pct", "map_entropy", "frontier_count",
                "path_length", "compute_time_ms", "control_smoothness",
            ])
            writer.writeheader()
            for s in run.steps:
                writer.writerow({
                    "step": s.step,
                    "time_s": f"{s.time_s:.4f}",
                    "pos_x": f"{s.pos_x:.4f}",
                    "pos_y": f"{s.pos_y:.4f}",
                    "pos_z": f"{s.pos_z:.4f}",
                    "coverage_pct": f"{s.coverage_pct:.6f}",
                    "map_entropy": f"{s.map_entropy:.4f}",
                    "frontier_count": s.frontier_count,
                    "path_length": f"{s.path_length:.4f}",
                    "compute_time_ms": f"{s.compute_time_ms:.3f}",
                    "control_smoothness": f"{s.control_smoothness:.6f}",
                })
        print(f"  Saved {fpath}")

    # Summary CSV
    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "variant", "env", "repeat",
            "final_coverage", "final_entropy", "total_path_length",
            "mean_compute_ms", "mean_smoothness",
        ])
        writer.writeheader()
        for run in all_metrics:
            last = run.steps[-1] if run.steps else StepMetrics()
            compute_times = [s.compute_time_ms for s in run.steps]
            smoothness = [s.control_smoothness for s in run.steps]
            writer.writerow({
                "variant": run.variant,
                "env": run.env,
                "repeat": run.repeat,
                "final_coverage": f"{last.coverage_pct:.6f}",
                "final_entropy": f"{last.map_entropy:.4f}",
                "total_path_length": f"{last.path_length:.4f}",
                "mean_compute_ms": f"{np.mean(compute_times):.3f}" if compute_times else "0",
                "mean_smoothness": f"{np.mean(smoothness):.6f}" if smoothness else "0",
            })
    print(f"  Summary: {summary_path}")


# ── CLI ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="I-MPPI Ablation Study")
    parser.add_argument(
        "--variants", nargs="+", default=["A0_full"],
        help="Variant names to run (e.g. A0_full A1_no_mi). Use --all for everything.",
    )
    parser.add_argument("--all", action="store_true", help="Run all variants")
    parser.add_argument(
        "--group", choices=["A", "B", "C"],
        help="Run all variants in a group",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--env", default="corridor", choices=["corridor", "warehouse"])
    parser.add_argument(
        "--output", type=str, default="results/ablation",
        help="Output directory for CSVs",
    )
    args = parser.parse_args()

    if args.all:
        variant_names = list(VARIANTS.keys())
    elif args.group:
        variant_names = [k for k in VARIANTS if k.startswith(args.group)]
    else:
        variant_names = args.variants

    # Validate
    for v in variant_names:
        if v not in VARIANTS:
            print(f"ERROR: Unknown variant '{v}'")
            print(f"Available: {', '.join(VARIANTS.keys())}")
            sys.exit(1)

    output_dir = Path(args.output)
    all_metrics: list[RunMetrics] = []

    print(f"I-MPPI Ablation: {len(variant_names)} variants × {args.repeats} repeats")
    print(f"Environment: {args.env}, Steps: {args.steps}")
    print()

    for vname in variant_names:
        overrides = VARIANTS[vname]
        for r in range(args.repeats):
            metrics = run_variant(vname, overrides, args.env, args.steps, r)
            all_metrics.append(metrics)

    save_metrics(all_metrics, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
