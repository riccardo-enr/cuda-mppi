"""
Quadrotor MPPI trajectory tracking test.

Reproduces the lemniscate (figure-8) tracking experiment from:

    Enrico, R., Mancini, M., & Capello, E. (2025). Comparison of NMPC and
    GPU-Parallelized MPPI for Real-Time UAV Control on Embedded Hardware.
    Applied Sciences, 15(16), 9114.

The original paper used ENU/FLU frames with JAX MPPI.  This test validates
the CUDA MPPI library (cuda_mppi) using the NED/FRD QuadrotorDynamics model.

Reference trajectory: 3D Bernoulli lemniscate (figure-8).
    x(t) = x_c + s * cos(θ) / (1 + sin²(θ))
    y(t) = y_c + s * sin(θ) cos(θ) / (1 + sin²(θ))
    z(t) = z_c                                          (NED: negative = up)
    θ = 2πt / T_period

Paper parameters (Table 4):
    K = 900, N = 40, dt = 0.02 s, λ = 1000
    Thrust ∈ [0, 40] N, roll/pitch rate ∈ [-3, 3] rad/s, yaw ∈ [-1.5, 1.5]
    Mass = 2.0 kg (Holybro X500)
"""

import argparse
import time

import cuda_mppi
import numpy as np

# ---------------------------------------------------------------------------
# Lemniscate reference generator
# ---------------------------------------------------------------------------


def lemniscate_reference(
    t: np.ndarray,
    centre: tuple[float, float, float] = (0.0, 0.0, -2.5),
    scale: float = 5.0,
    period: float = 15.0,
) -> np.ndarray:
    """Generate a 3D Bernoulli lemniscate in NED frame.

    Returns an (len(t), 3) array of [px, py, pz] positions.
    """
    theta = 2.0 * np.pi * t / period
    denom = 1.0 + np.sin(theta) ** 2
    x = centre[0] + scale * np.cos(theta) / denom
    y = centre[1] + scale * np.sin(theta) * np.cos(theta) / denom
    z = np.full_like(t, centre[2])
    return np.column_stack([x, y, z]).astype(np.float32)


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------


def run_tracking(
    num_samples: int = 900,
    horizon: int = 40,
    dt: float = 0.02,
    lam: float = 1000.0,
    sim_time: float = 30.0,
    mass: float = 2.0,
    plot: bool = False,
    diagnose: bool = False,
):
    """Run closed-loop MPPI trajectory tracking and compute RMSE."""

    # --- Reference trajectory (full simulation) ---
    n_steps = int(sim_time / dt)
    t_vec = np.arange(n_steps) * dt
    ref_full = lemniscate_reference(t_vec)

    # --- MPPI configuration ---
    config = cuda_mppi.MPPIConfig(
        num_samples=num_samples,
        horizon=horizon,
        nx=13,
        nu=4,
        lambda_=lam,
        dt=dt,
    )
    # Exploration noise std: [thrust, roll_rate, pitch_rate, yaw_rate]
    config.set_control_sigma([2.0, 0.5, 0.5, 0.3])

    # --- Dynamics ---
    dynamics = cuda_mppi.QuadrotorDynamics()
    dynamics.mass = mass
    dynamics.gravity = 9.81
    dynamics.tau_omega = 0.05

    # --- Cost (quadratic tracking, Eq. 7 from paper) ---
    cost = cuda_mppi.TrackingCost()
    cost.set_Q_pos([5e3, 5e3, 5e3])
    cost.set_Q_vel([4e1, 4e1, 4e1])
    cost.Q_quat = 2e1
    cost.set_Q_omega([2e1, 2e1, 2e1])
    cost.set_R([2e-2, 2e-1, 2e-1, 2e-1])
    cost.set_R_delta([1e-3, 1e-2, 1e-2, 1e-2])
    cost.terminal_weight = 10.0
    cost.set_hover_pos([0.0, 0.0, -2.5])

    # --- Controller ---
    controller = cuda_mppi.QuadrotorMPPI(config, dynamics, cost)

    # Initialise nominal control at hover
    hover_thrust = mass * 9.81  # T = mg for NED hover
    u_hover = np.array([hover_thrust, 0.0, 0.0, 0.0], dtype=np.float32)
    controller.set_nominal_control(u_hover)

    # --- Initial state (NED) ---
    # [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    state = np.zeros(13, dtype=np.float32)
    state[0] = 2.0  # px = 2 m North
    state[1] = 2.0  # py = 2 m East
    state[2] = 0.0  # pz = 0  (ground)
    state[6] = 1.0  # qw = 1  (identity quaternion)

    # --- Storage ---
    positions = np.zeros((n_steps, 3), dtype=np.float32)
    controls = np.zeros((n_steps, 4), dtype=np.float32)
    comp_times = np.zeros(n_steps, dtype=np.float64)
    cost_stds: list[float] = []
    cost_means: list[float] = []
    cost_mins: list[float] = []

    # --- Simulation ---
    print(
        f"Running MPPI tracking: K={num_samples}, N={horizon}, "
        f"dt={dt}, λ={lam}, T_sim={sim_time}s"
    )

    for k in range(n_steps):
        positions[k] = state[:3]

        # Build full state reference for the horizon window.
        # TrackingCost expects [horizon × 13]: pos, vel=0, quat=[1,0,0,0], omega=0
        ref_start = k
        ref_end = min(k + horizon, n_steps)
        pos_window = ref_full[ref_start:ref_end]
        if len(pos_window) < horizon:
            pad = np.tile(pos_window[-1:], (horizon - len(pos_window), 1))
            pos_window = np.concatenate([pos_window, pad], axis=0)
        state_ref = np.zeros((horizon, 13), dtype=np.float32)
        state_ref[:, :3] = pos_window  # position
        state_ref[:, 6] = 1.0  # qw = 1 (level attitude)
        controller.set_state_reference(state_ref.flatten(), horizon)

        # Feed back previous action for rate-of-change cost
        if k > 0:
            controller.set_applied_control(controls[k - 1])

        # Shift + compute
        controller.shift()
        t0 = time.perf_counter()
        controller.compute(state)
        comp_times[k] = (time.perf_counter() - t0) * 1e3  # ms

        action = controller.get_action()
        controls[k] = action

        if diagnose and k % 10 == 0:
            costs = np.array(controller.get_last_costs())
            cost_stds.append(costs.std())
            cost_means.append(costs.mean())
            cost_mins.append(costs.min())

        # Propagate plant dynamics (host-side RK4)
        state = dynamics.step(state, action, dt)

        # Progress
        if (k + 1) % 500 == 0 or k == 0:
            pos_err = np.linalg.norm(positions[k] - ref_full[k])
            print(
                f"  step {k + 1:5d}/{n_steps}  |  "
                f"pos_err={pos_err:.3f} m  |  "
                f"comp={comp_times[k]:.2f} ms"
            )

    # --- Metrics ---
    errors = positions - ref_full
    rmse_xyz = np.sqrt(np.mean(errors**2, axis=0))
    rmse_total = np.sqrt(np.mean(np.sum(errors**2, axis=1)))

    print("\n--- Results ---")
    print(f"RMSE  X: {rmse_xyz[0]:.4f} m")
    print(f"RMSE  Y: {rmse_xyz[1]:.4f} m")
    print(f"RMSE  Z: {rmse_xyz[2]:.4f} m")
    print(f"RMSE total: {rmse_total:.4f} m")
    print(
        f"Compute time — mean: {np.mean(comp_times):.2f} ms, "
        f"median: {np.median(comp_times):.2f} ms, "
        f"p95: {np.percentile(comp_times, 95):.2f} ms"
    )

    if diagnose and cost_stds:
        c_std_med = float(np.median(cost_stds))
        print("\n--- Cost diagnostics ---")
        print(
            f"  rollout cost std  — median: {c_std_med:.1f},  mean: {np.mean(cost_stds):.1f}"
        )
        print(f"  rollout cost mean — median: {np.median(cost_means):.1f}")
        print(f"  rollout cost min  — median: {np.median(cost_mins):.1f}")
        print()
        print("  c_std is how much rollouts differ; target c_std/λ in [1, 10].")
        print(f"  Current λ={lam}  →  c_std/λ = {c_std_med / lam:.2f}")
        print()
        magnitude = 10 ** int(np.log10(max(c_std_med, 1)))
        candidates = sorted(
            set(
                int(magnitude * f)
                for f in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
                if int(magnitude * f) > 0
            )
        )
        best_lam, best_score = lam, -1.0
        for lam_c in candidates:
            ratio = c_std_med / lam_c
            good = 1.0 <= ratio <= 10.0
            flag = "  ← good" if good else ""
            score = 1.0 / abs(ratio - 3.0) if good else 0.0  # prefer ratio≈3
            if good and score > best_score:
                best_score, best_lam = score, lam_c
            print(f"  λ = {lam_c:>8}  |  c_std/λ = {ratio:6.2f}{flag}")
        print(
            f"\n  Suggested λ ≈ {best_lam}  (c_std/λ = {c_std_med / best_lam:.2f})"
        )

    if plot:
        _plot_results(
            t_vec, positions, ref_full, controls, comp_times, rmse_total
        )

    return rmse_total


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_results(t, pos, ref, ctrl, comp_t, rmse):
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    plt.style.use(["science", "ieee"])
    plt.rcParams["figure.dpi"] = 150

    # --- 3D trajectory ---
    fig = plt.figure(figsize=(3.3, 3.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        ref[:, 0],
        ref[:, 1],
        -ref[:, 2],  # NED→display: negate z
        "r--",
        lw=0.8,
        label="Reference",
    )
    ax.plot(pos[:, 0], pos[:, 1], -pos[:, 2], "b-", lw=0.6, label="MPPI")
    ax.scatter(
        *pos[0, :2], -pos[0, 2], c="green", s=20, zorder=5, label="Start"
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"MPPI Lemniscate Tracking — RMSE: {rmse:.3f} m")
    ax.legend(fontsize=6)
    fig.savefig("tracking_3d.png", dpi=300, bbox_inches="tight")

    # --- Per-axis position tracking ---
    labels = ["X (m)", "Y (m)", "Z (m)"]
    fig, axes = plt.subplots(3, 1, figsize=(3.3, 4.0), constrained_layout=True)
    for i, (ax, lab) in enumerate(zip(axes, labels)):
        ref_plot = ref[:, i] if i < 2 else -ref[:, i]
        pos_plot = pos[:, i] if i < 2 else -pos[:, i]
        ax.plot(t, ref_plot, "r--", lw=0.8, label="Ref")
        ax.plot(t, pos_plot, "b-", lw=0.6, label="MPPI")
        ax.set_ylabel(lab)
        if i == 0:
            ax.legend(fontsize=6)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Position Tracking")
    fig.savefig("tracking_axes.png", dpi=300, bbox_inches="tight")

    # --- Control inputs ---
    ctrl_labels = [
        "Thrust (N)",
        "Roll rate (rad/s)",
        "Pitch rate (rad/s)",
        "Yaw rate (rad/s)",
    ]
    fig, axes = plt.subplots(4, 1, figsize=(3.3, 5.0), constrained_layout=True)
    for i, (ax, lab) in enumerate(zip(axes, ctrl_labels)):
        ax.plot(t, ctrl[:, i], "b-", lw=0.5)
        ax.set_ylabel(lab)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("MPPI Control Inputs")
    fig.savefig("tracking_controls.png", dpi=300, bbox_inches="tight")

    # --- Computation time ---
    fig, ax = plt.subplots(figsize=(3.3, 2.0), constrained_layout=True)
    ax.plot(t, comp_t, "b-", lw=0.4, alpha=0.6)
    ax.axhline(20.0, color="r", ls="--", lw=0.8, label="20 ms budget")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Compute time (ms)")
    ax.set_title("MPPI Computation Time")
    ax.legend(fontsize=6)
    fig.savefig("tracking_compute.png", dpi=300, bbox_inches="tight")

    plt.show()


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------


def test_quadrotor_tracking():
    """Verify MPPI tracks a lemniscate with RMSE < 2.0 m (short sim)."""
    rmse = run_tracking(sim_time=5.0, plot=False)
    assert rmse < 2.0, f"RMSE {rmse:.3f} m exceeds 2.0 m threshold"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MPPI quadrotor lemniscate tracking test"
    )
    parser.add_argument(
        "--samples",
        "-K",
        type=int,
        default=900,
        help="Number of MPPI samples (default: 900)",
    )
    parser.add_argument(
        "--horizon",
        "-N",
        type=int,
        default=40,
        help="Prediction horizon steps (default: 40)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Control timestep (default: 0.02)",
    )
    parser.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=1000.0,
        help="Temperature parameter (default: 1000)",
    )
    parser.add_argument(
        "--time",
        "-T",
        type=float,
        default=30.0,
        help="Simulation time in seconds (default: 30)",
    )
    parser.add_argument(
        "--mass", type=float, default=2.0, help="UAV mass in kg (default: 2.0)"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Show matplotlib plots"
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print cost-scale statistics after the run to guide lambda tuning",
    )
    args = parser.parse_args()

    rmse = run_tracking(
        num_samples=args.samples,
        horizon=args.horizon,
        dt=args.dt,
        lam=args.lam,
        sim_time=args.time,
        mass=args.mass,
        plot=args.plot,
        diagnose=args.diagnose,
    )

    # Sanity check: paper achieved 0.69 m RMSE with full cost.
    # Position-only cost is coarser, so we allow up to 2.0 m.
    assert rmse < 2.0, f"RMSE {rmse:.3f} m exceeds 2.0 m threshold"
    print("\n✓ Tracking test passed.")
