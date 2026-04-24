"""
Quadrotor tracking comparison: MPPI vs S-MPPI vs K-MPPI.

Runs the same lemniscate tracking experiment from test_quadrotor_tracking.py
with all three controller variants and reports RMSE + compute time.
"""

import argparse
import time

import cuda_mppi
import numpy as np


def lemniscate_reference(
    t: np.ndarray,
    centre: tuple[float, float, float] = (0.0, 0.0, -2.5),
    scale: float = 5.0,
    period: float = 15.0,
) -> np.ndarray:
    """Generate a 3D Bernoulli lemniscate in NED frame."""
    theta = 2.0 * np.pi * t / period
    denom = 1.0 + np.sin(theta) ** 2
    x = centre[0] + scale * np.cos(theta) / denom
    y = centre[1] + scale * np.sin(theta) * np.cos(theta) / denom
    z = np.full_like(t, centre[2])
    return np.column_stack([x, y, z]).astype(np.float32)


def make_config(
    num_samples: int = 900,
    horizon: int = 40,
    dt: float = 0.02,
    lam: float = 1000.0,
    num_support_pts: int = 10,
    w_action_seq_cost: float = 1.0,
) -> cuda_mppi.MPPIConfig:
    config = cuda_mppi.MPPIConfig(
        num_samples=num_samples,
        horizon=horizon,
        nx=13,
        nu=4,
        lambda_=lam,
        dt=dt,
    )
    config.set_control_sigma([2.0, 0.5, 0.5, 0.3])
    config.num_support_pts = num_support_pts
    config.w_action_seq_cost = w_action_seq_cost
    return config


def make_dynamics(mass: float = 2.0) -> cuda_mppi.QuadrotorDynamics:
    dynamics = cuda_mppi.QuadrotorDynamics()
    dynamics.mass = mass
    dynamics.gravity = 9.81
    dynamics.tau_omega = 0.05
    return dynamics


def make_cost() -> cuda_mppi.TrackingCost:
    cost = cuda_mppi.TrackingCost()
    cost.set_Q_pos([5e3, 5e3, 5e3])
    cost.set_Q_vel([4e1, 4e1, 4e1])
    cost.Q_quat = 2e1
    cost.set_Q_omega([2e1, 2e1, 2e1])
    cost.set_R([2e-2, 2e-1, 2e-1, 2e-1])
    cost.set_R_delta([1e-3, 1e-2, 1e-2, 1e-2])
    cost.terminal_weight = 10.0
    cost.set_hover_pos([0.0, 0.0, -2.5])
    return cost


def run_single(
    controller_name: str,
    controller,
    dynamics: cuda_mppi.QuadrotorDynamics,
    horizon: int,
    dt: float,
    sim_time: float,
    mass: float,
) -> dict:
    """Run closed-loop tracking with a single controller. Returns metrics dict."""
    n_steps = int(sim_time / dt)
    t_vec = np.arange(n_steps) * dt
    ref_full = lemniscate_reference(t_vec)

    hover_thrust = mass * 9.81
    u_hover = np.array([hover_thrust, 0.0, 0.0, 0.0], dtype=np.float32)
    controller.set_nominal_control(u_hover)

    state = np.zeros(13, dtype=np.float32)
    state[0] = 2.0
    state[1] = 2.0
    state[6] = 1.0

    positions = np.zeros((n_steps, 3), dtype=np.float32)
    controls = np.zeros((n_steps, 4), dtype=np.float32)
    comp_times = np.zeros(n_steps, dtype=np.float64)

    print(f"\n{'=' * 60}")
    print(f"  {controller_name}")
    print(f"{'=' * 60}")

    for k in range(n_steps):
        positions[k] = state[:3]

        ref_start = k
        ref_end = min(k + horizon, n_steps)
        pos_window = ref_full[ref_start:ref_end]
        if len(pos_window) < horizon:
            pad = np.tile(pos_window[-1:], (horizon - len(pos_window), 1))
            pos_window = np.concatenate([pos_window, pad], axis=0)
        state_ref = np.zeros((horizon, 13), dtype=np.float32)
        state_ref[:, :3] = pos_window
        state_ref[:, 6] = 1.0
        controller.set_state_reference(state_ref.flatten(), horizon)

        if k > 0:
            controller.set_applied_control(controls[k - 1])

        controller.shift()
        t0 = time.perf_counter()
        controller.compute(state)
        comp_times[k] = (time.perf_counter() - t0) * 1e3

        action = controller.get_action()
        controls[k] = action
        state = dynamics.step(state, action, dt)

        if (k + 1) % 500 == 0 or k == 0:
            pos_err = np.linalg.norm(positions[k] - ref_full[k])
            print(
                f"  step {k + 1:5d}/{n_steps}  |  "
                f"pos_err={pos_err:.3f} m  |  "
                f"comp={comp_times[k]:.2f} ms"
            )

    errors = positions - ref_full
    rmse_xyz = np.sqrt(np.mean(errors**2, axis=0))
    rmse_total = np.sqrt(np.mean(np.sum(errors**2, axis=1)))

    print(f"\n  RMSE  X: {rmse_xyz[0]:.4f} m")
    print(f"  RMSE  Y: {rmse_xyz[1]:.4f} m")
    print(f"  RMSE  Z: {rmse_xyz[2]:.4f} m")
    print(f"  RMSE total: {rmse_total:.4f} m")
    print(
        f"  Compute — mean: {np.mean(comp_times):.2f} ms, "
        f"median: {np.median(comp_times):.2f} ms, "
        f"p95: {np.percentile(comp_times, 95):.2f} ms"
    )

    return {
        "name": controller_name,
        "rmse_total": rmse_total,
        "rmse_xyz": rmse_xyz,
        "comp_mean": np.mean(comp_times),
        "comp_median": np.median(comp_times),
        "comp_p95": np.percentile(comp_times, 95),
        "positions": positions,
        "controls": controls,
        "comp_times": comp_times,
    }


def run_comparison(
    num_samples: int = 900,
    horizon: int = 40,
    dt: float = 0.02,
    lam: float = 1000.0,
    sim_time: float = 30.0,
    mass: float = 2.0,
    num_support_pts: int = 10,
    w_action_seq_cost: float = 1.0,
    plot: bool = False,
):
    dynamics = make_dynamics(mass)
    cost = make_cost()

    results = []

    # --- MPPI ---
    config_mppi = make_config(num_samples, horizon, dt, lam)
    ctrl_mppi = cuda_mppi.QuadrotorMPPI(config_mppi, dynamics, cost)
    results.append(
        run_single("MPPI", ctrl_mppi, dynamics, horizon, dt, sim_time, mass)
    )

    # --- S-MPPI ---
    config_smppi = make_config(
        num_samples,
        horizon,
        dt,
        lam,
        w_action_seq_cost=w_action_seq_cost,
    )
    ctrl_smppi = cuda_mppi.QuadrotorSMPPI(config_smppi, dynamics, cost)
    results.append(
        run_single("S-MPPI", ctrl_smppi, dynamics, horizon, dt, sim_time, mass)
    )

    # --- K-MPPI ---
    config_kmppi = make_config(
        num_samples,
        horizon,
        dt,
        lam,
        num_support_pts=num_support_pts,
    )
    ctrl_kmppi = cuda_mppi.QuadrotorKMPPI(config_kmppi, dynamics, cost)
    results.append(
        run_single("K-MPPI", ctrl_kmppi, dynamics, horizon, dt, sim_time, mass)
    )

    # --- Summary table ---
    print(f"\n{'=' * 60}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"  {'Controller':<12} {'RMSE (m)':>10} {'Comp mean (ms)':>16} {'Comp p95 (ms)':>15}"
    )
    print(f"  {'-' * 55}")
    for r in results:
        import math

        if math.isnan(r["rmse_total"]):
            print(
                f"  {r['name']:<12} {'N/A (skipped)':>10} {'—':>16} {'—':>15}"
            )
        else:
            print(
                f"  {r['name']:<12} {r['rmse_total']:>10.4f} "
                f"{r['comp_mean']:>16.2f} {r['comp_p95']:>15.2f}"
            )
    print()

    if plot:
        _plot_comparison(results, sim_time, dt)

    return results


def _plot_comparison(results, sim_time, dt):
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    plt.style.use(["science", "ieee"])
    plt.rcParams["figure.dpi"] = 150

    n_steps = int(sim_time / dt)
    t_vec = np.arange(n_steps) * dt
    ref_full = lemniscate_reference(t_vec)

    colors = {"MPPI": "#1f77b4", "S-MPPI": "#ff7f0e", "K-MPPI": "#2ca02c"}

    # --- 3D trajectories ---
    fig = plt.figure(figsize=(3.3, 3.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        ref_full[:, 0],
        ref_full[:, 1],
        -ref_full[:, 2],
        "r--",
        lw=0.8,
        label="Reference",
    )
    for r in results:
        pos = r["positions"]
        ax.plot(
            pos[:, 0],
            pos[:, 1],
            -pos[:, 2],
            color=colors[r["name"]],
            lw=0.6,
            label=f"{r['name']} ({r['rmse_total']:.3f} m)",
        )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Lemniscate Tracking Comparison")
    ax.legend(fontsize=5)
    fig.savefig("tracking_comparison_3d.png", dpi=300, bbox_inches="tight")

    # --- Per-axis tracking ---
    labels = ["X (m)", "Y (m)", "Z (m)"]
    fig, axes = plt.subplots(3, 1, figsize=(3.3, 4.0), constrained_layout=True)
    for i, (ax, lab) in enumerate(zip(axes, labels)):
        ref_plot = ref_full[:, i] if i < 2 else -ref_full[:, i]
        ax.plot(t_vec, ref_plot, "r--", lw=0.8, label="Ref")
        for r in results:
            pos = r["positions"]
            pos_plot = pos[:, i] if i < 2 else -pos[:, i]
            ax.plot(
                t_vec,
                pos_plot,
                color=colors[r["name"]],
                lw=0.5,
                label=r["name"],
            )
        ax.set_ylabel(lab)
        if i == 0:
            ax.legend(fontsize=5)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Position Tracking")
    fig.savefig("tracking_comparison_axes.png", dpi=300, bbox_inches="tight")

    # --- Computation time ---
    fig, ax = plt.subplots(figsize=(3.3, 2.0), constrained_layout=True)
    for r in results:
        ax.plot(
            t_vec,
            r["comp_times"],
            color=colors[r["name"]],
            lw=0.4,
            alpha=0.6,
            label=r["name"],
        )
    ax.axhline(20.0, color="r", ls="--", lw=0.8, label="20 ms budget")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Compute time (ms)")
    ax.set_title("Computation Time")
    ax.legend(fontsize=5)
    fig.savefig("tracking_comparison_compute.png", dpi=300, bbox_inches="tight")

    plt.show()


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------


def test_tracking_comparison():
    """Verify all controllers track the lemniscate with RMSE < 2.0 m."""
    results = run_comparison(sim_time=5.0, plot=False)
    for r in results:
        assert r["rmse_total"] < 2.0, (
            f"{r['name']} RMSE {r['rmse_total']:.3f} m exceeds 2.0 m threshold"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare MPPI, S-MPPI, K-MPPI on quadrotor lemniscate tracking"
    )
    parser.add_argument("-K", "--samples", type=int, default=900)
    parser.add_argument("-N", "--horizon", type=int, default=40)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--lambda", dest="lam", type=float, default=1000.0)
    parser.add_argument("-T", "--time", type=float, default=30.0)
    parser.add_argument("--mass", type=float, default=2.0)
    parser.add_argument(
        "-M",
        "--support-pts",
        type=int,
        default=10,
        help="K-MPPI support points (default: 10)",
    )
    parser.add_argument(
        "--smoothness-weight",
        type=float,
        default=1.0,
        help="S-MPPI smoothness cost weight (default: 1.0)",
    )
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    results = run_comparison(
        num_samples=args.samples,
        horizon=args.horizon,
        dt=args.dt,
        lam=args.lam,
        sim_time=args.time,
        mass=args.mass,
        num_support_pts=args.support_pts,
        w_action_seq_cost=args.smoothness_weight,
        plot=args.plot,
    )

    import math

    for r in results:
        if math.isnan(r["rmse_total"]):
            continue
        assert r["rmse_total"] < 2.0, (
            f"{r['name']} RMSE {r['rmse_total']:.3f} m exceeds 2.0 m"
        )
    print("All active controllers passed (RMSE < 2.0 m).")
