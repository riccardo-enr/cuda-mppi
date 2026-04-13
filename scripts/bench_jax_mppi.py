"""
JAX vs CUDA MPPI speed benchmark.  refs #26

Implements the same MPPI algorithm as the CUDA library using JAX/vmap/jit and
measures wall-clock time per iteration for a range of sample counts.  The CUDA
baseline uses cuda_mppi.QuadrotorMPPI with identical parameters so the
comparison is apples-to-apples (same algorithm, same GPU, different stack).

Quadrotor dynamics and tracking cost are re-implemented to match exactly:
    include/mppi/instantiations/quadrotor.cuh
    include/mppi/instantiations/tracking_cost.cuh

MPPI algorithm matches:
    include/mppi/core/softmax.cuh
    include/mppi/core/kernels.cuh
    include/mppi/controllers/mppi.cuh  (single-iteration additive update)
"""

from __future__ import annotations

import argparse
import time

import cuda_mppi
import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Quadrotor parameters (must match QuadrotorDynamics defaults / test config)
# ---------------------------------------------------------------------------

MASS = 2.0
GRAVITY = 9.81
TAU_OMEGA = 0.05
DT = 0.02
HORIZON = 40
LAMBDA = 10000.0
SIGMA = np.array([2.0, 0.5, 0.5, 0.3], dtype=np.float32)

U_MIN = np.array([0.0, -10.0, -10.0, -10.0], dtype=np.float32)
U_MAX = np.array([4.0 * GRAVITY, 10.0, 10.0, 10.0], dtype=np.float32)

# Cost weights — identical to TrackingCost defaults (tracking_cost.cuh lines 45-63)
Q_POS = jnp.array([5e3, 5e3, 5e3])
Q_VEL = jnp.array([4e1, 4e1, 4e1])
Q_QUAT = 2e1
Q_OMEGA = jnp.array([2e1, 2e1, 2e1])
R_CTRL = jnp.array([2e-2, 2e-1, 2e-1, 2e-1])
R_DELTA = jnp.array([1e-3, 1e-2, 1e-2, 1e-2])
TERMINAL_WEIGHT = 10.0


# ---------------------------------------------------------------------------
# JAX quadrotor dynamics (NED / FRD, matches quadrotor.cuh)
# ---------------------------------------------------------------------------


def _quat_rotate(q: jax.Array, v: jax.Array) -> jax.Array:
    """Rotate vector v by unit quaternion q = [qw, qx, qy, qz] (active rotation).

    Implements the same rotation matrix as quat_rotate() in quadrotor.cuh
    (lines 68-87).
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    R = jnp.array([
        [
            1 - 2 * (qy**2 + qz**2),
            2 * (qx * qy - qw * qz),
            2 * (qx * qz + qw * qy),
        ],
        [
            2 * (qx * qy + qw * qz),
            1 - 2 * (qx**2 + qz**2),
            2 * (qy * qz - qw * qx),
        ],
        [
            2 * (qx * qz - qw * qy),
            2 * (qy * qz + qw * qx),
            1 - 2 * (qx**2 + qy**2),
        ],
    ])
    return R @ v


def _state_deriv(x: jax.Array, u: jax.Array) -> jax.Array:
    """Continuous-time state derivative.  Matches state_deriv() in quadrotor.cuh
    (lines 96-126).

    x = [px,py,pz, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz]  (13-D)
    u = [T, wx_cmd, wy_cmd, wz_cmd]                     (4-D)
    """
    p = x[0:3]  # noqa: F841
    v = x[3:6]
    q = x[6:10]
    omega = x[10:13]

    T = u[0]
    omega_cmd = u[1:4]

    # --- position derivative ---
    dp = v

    # --- velocity derivative ---
    # Thrust in FRD body frame: [0, 0, -T]
    thrust_body = jnp.array([0.0, 0.0, -T])
    thrust_world = _quat_rotate(q, thrust_body)
    # NED: gravity is +z direction
    dv = thrust_world / MASS + jnp.array([0.0, 0.0, GRAVITY])

    # --- quaternion derivative: q_dot = 0.5 * q ⊗ [0, ω] ---
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    wx, wy, wz = omega[0], omega[1], omega[2]
    dqw = 0.5 * (-wx * qx - wy * qy - wz * qz)
    dqx = 0.5 * (wx * qw + wz * qy - wy * qz)
    dqy = 0.5 * (wy * qw - wz * qx + wx * qz)
    dqz = 0.5 * (wz * qw + wy * qx - wx * qy)
    dq = jnp.array([dqw, dqx, dqy, dqz])

    # --- angular rate derivative: first-order lag ---
    domega = (omega_cmd - omega) / TAU_OMEGA

    return jnp.concatenate([dp, dv, dq, domega])


def _rk4_step(x: jax.Array, u: jax.Array, dt: float) -> jax.Array:
    """RK4 integration + quaternion normalisation.  Matches step() in
    quadrotor.cuh (lines 139-183).
    """
    k1 = _state_deriv(x, u)
    k2 = _state_deriv(x + 0.5 * dt * k1, u)
    k3 = _state_deriv(x + 0.5 * dt * k2, u)
    k4 = _state_deriv(x + dt * k3, u)
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Normalise quaternion (indices 6-10)
    q_norm = jnp.linalg.norm(x_next[6:10]) + 1e-8
    x_next = x_next.at[6:10].set(x_next[6:10] / q_norm)
    return x_next


# ---------------------------------------------------------------------------
# JAX tracking cost (matches tracking_cost.cuh)
# ---------------------------------------------------------------------------


def _running_cost(
    x: jax.Array,
    u: jax.Array,
    u_prev: jax.Array,
    x_ref: jax.Array,
) -> jax.Array:
    """Per-timestep running cost.  Matches compute() in tracking_cost.cuh
    (lines 81-138).
    """
    pos_err = x[0:3] - x_ref[0:3]
    vel_err = x[3:6] - x_ref[3:6]
    q = x[6:10]
    q_ref = x_ref[6:10]
    omega_err = x[10:13] - x_ref[10:13]

    # Quaternion distance: 1 - <q, q_ref>^2
    quat_dist = 1.0 - jnp.dot(q, q_ref) ** 2

    c = (
        jnp.sum(Q_POS * pos_err**2)
        + jnp.sum(Q_VEL * vel_err**2)
        + Q_QUAT * quat_dist
        + jnp.sum(Q_OMEGA * omega_err**2)
        + jnp.sum(R_CTRL * u**2)
        + jnp.sum(R_DELTA * (u - u_prev) ** 2)
    )
    return c


def _terminal_cost(x: jax.Array, x_ref: jax.Array) -> jax.Array:
    """Terminal cost.  Matches terminal_cost() in tracking_cost.cuh
    (lines 143-171) — position, velocity, quaternion only (no ω, no control).
    """
    pos_err = x[0:3] - x_ref[0:3]
    vel_err = x[3:6] - x_ref[3:6]
    q = x[6:10]
    q_ref = x_ref[6:10]
    quat_dist = 1.0 - jnp.dot(q, q_ref) ** 2

    c = TERMINAL_WEIGHT * (
        jnp.sum(Q_POS * pos_err**2)
        + jnp.sum(Q_VEL * vel_err**2)
        + Q_QUAT * quat_dist
    )
    return c


# ---------------------------------------------------------------------------
# JAX MPPI step (single-iteration additive update)
# ---------------------------------------------------------------------------


def _rollout_single(
    x0: jax.Array,
    u_nom: jax.Array,
    eps: jax.Array,
    sigma: jax.Array,
    state_ref: jax.Array,
    u_prev_init: jax.Array,
) -> jax.Array:
    """Simulate one trajectory and return total cost.

    u_nom : (horizon, nu)
    eps   : (horizon, nu)  — standard-normal noise
    sigma : (nu,)
    state_ref : (horizon, nx)
    u_prev_init : (nu,)    — control applied before the horizon

    Returns scalar total cost.
    """
    u_min = jnp.array(U_MIN)
    u_max = jnp.array(U_MAX)

    def step_fn(carry, t):
        x, u_prev, total_cost = carry
        u_k = jnp.clip(u_nom[t] + eps[t] * sigma, u_min, u_max)
        c = _running_cost(x, u_k, u_prev, state_ref[t])
        x_next = _rk4_step(x, u_k, DT)
        return (x_next, u_k, total_cost + c), None

    (x_T, _, total_cost), _ = jax.lax.scan(
        step_fn,
        (x0, u_prev_init, jnp.zeros(())),
        jnp.arange(u_nom.shape[0]),
    )
    total_cost = total_cost + _terminal_cost(x_T, state_ref[-1])
    return total_cost


# vmap over K samples
_rollout_batch = jax.vmap(
    _rollout_single,
    in_axes=(None, None, 0, None, None, None),
)


def make_mppi_step(num_samples: int, horizon: int, lam: float):
    """Factory that returns a JIT-compiled MPPI step function for a fixed K.

    The returned function has signature:
        step(key, x0, u_nom, u_prev, state_ref) -> u_nom_updated
    """
    sigma = jnp.array(SIGMA)

    @jax.jit
    def mppi_step(
        key: jax.Array,
        x0: jax.Array,
        u_nom: jax.Array,
        u_prev: jax.Array,
        state_ref: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Returns (u_nom_updated, costs) where costs has shape (K,)."""
        # Sample noise: (K, horizon, nu)
        eps = jax.random.normal(key, shape=(num_samples, horizon, len(SIGMA)))

        # Rollout all K trajectories in parallel
        costs = _rollout_batch(x0, u_nom, eps, sigma, state_ref, u_prev)

        # Softmax weights (numerically stable)
        costs_shifted = costs - jnp.min(costs)
        w = jnp.exp(-costs_shifted / lam)
        w = w / jnp.sum(w)  # (K,)

        # Weighted additive update
        # delta_u[t,i] = sum_k w_k * eps_k[t,i] * sigma[i]
        delta_u = jnp.einsum("k,kti,i->ti", w, eps, sigma)
        u_nom_new = u_nom + delta_u
        return u_nom_new, costs

    return mppi_step


# ---------------------------------------------------------------------------
# Cost-scale diagnostic (for lambda tuning)
# ---------------------------------------------------------------------------


def diagnose_costs(num_samples: int = 900) -> None:
    """Sample one batch of rollouts and print cost statistics.

    Per the tuning guide, the right λ satisfies:
        0.1 < (costs.mean() - costs.min()) / lambda < 10

    Run this before setting lambda_ to get the right order of magnitude.
    """
    sigma = jnp.array(SIGMA)
    key = jax.random.PRNGKey(42)

    x0 = np.zeros(13, dtype=np.float32)
    x0[0] = 2.0
    x0[1] = 2.0
    x0[6] = 1.0  # same initial state as tracking test

    ref = jnp.array(_lemniscate_ref(HORIZON))
    u_nom = (
        jnp.zeros((HORIZON, 4), dtype=jnp.float32).at[:, 0].set(MASS * GRAVITY)
    )
    u_prev = jnp.zeros(4, dtype=jnp.float32)
    x0_j = jnp.array(x0)

    eps = jax.random.normal(key, shape=(num_samples, HORIZON, len(SIGMA)))
    costs = _rollout_batch(x0_j, u_nom, eps, sigma, ref, u_prev)
    costs = np.array(costs)

    c_min = costs.min()
    c_max = costs.max()
    c_mean = costs.mean()
    c_std = costs.std()

    print(f"\n--- Cost diagnostics (K={num_samples}, N={HORIZON}, dt={DT}) ---")
    print(f"  min  : {c_min:.1f}")
    print(f"  mean : {c_mean:.1f}")
    print(f"  max  : {c_max:.1f}")
    print(f"  std  : {c_std:.1f}  ← key quantity: how much rollouts differ")
    print()
    print(
        "  Absolute cost level is irrelevant (dominated by constant initial error)."
    )
    print(
        "  Target: c_std / λ in [1, 10]. Below 1 → uniform weights. Above 10 → collapse."
    )
    print()

    # Sweep candidates around c_std
    magnitude = 10 ** int(np.log10(max(c_std, 1)))
    candidates = sorted(
        set([
            int(magnitude * f)
            for f in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        ])
    )
    candidates = [c for c in candidates if c > 0]

    best_lam, best_score = candidates[0], -1.0
    for lam in candidates:
        w = np.exp(-(costs - c_min) / lam)
        w /= w.sum()
        eff_k = 1.0 / np.sum(w**2)
        ratio = c_std / lam
        good = 1.0 <= ratio <= 10.0
        flag = "  ← good" if good else ""
        print(
            f"  λ = {lam:>8}  |  c_std/λ = {ratio:6.2f}  |  eff. samples = {eff_k:6.1f} / {num_samples}{flag}"
        )
        # Prefer eff_k in 5–30% of K: sharp enough to be useful, not collapsed
        score = (
            eff_k
            if (0.05 * num_samples <= eff_k <= 0.30 * num_samples)
            else 0.0
        )
        if good and score > best_score:
            best_score, best_lam = score, lam

    print()
    print(
        f"  Suggested starting λ ≈ {best_lam}  (c_std/λ = {c_std / best_lam:.2f})"
    )
    print()


# ---------------------------------------------------------------------------
# Reference trajectory helper
# ---------------------------------------------------------------------------


def _lemniscate_ref(
    horizon: int,
    t0: float = 0.0,
    dt: float = DT,
    centre: tuple = (0.0, 0.0, -2.5),
    scale: float = 5.0,
    period: float = 15.0,
) -> np.ndarray:
    """Return a (horizon, 13) state reference window starting at t0."""
    t = t0 + np.arange(horizon) * dt
    theta = 2.0 * np.pi * t / period
    denom = 1.0 + np.sin(theta) ** 2
    x = centre[0] + scale * np.cos(theta) / denom
    y = centre[1] + scale * np.sin(theta) * np.cos(theta) / denom
    z = np.full_like(t, centre[2])

    ref = np.zeros((horizon, 13), dtype=np.float32)
    ref[:, 0] = x
    ref[:, 1] = y
    ref[:, 2] = z
    ref[:, 6] = 1.0  # qw = 1 (identity quaternion)
    return ref


# ---------------------------------------------------------------------------
# Closed-loop tracking (validates correctness of the JAX implementation)
# ---------------------------------------------------------------------------


def run_tracking(
    num_samples: int = 900,
    sim_time: float = 30.0,
    plot: bool = False,
    diagnose: bool = False,
) -> float:
    """Run JAX MPPI in closed-loop on the lemniscate and return RMSE.

    Mirrors test_quadrotor_tracking.py so results are directly comparable.
    Plant dynamics are propagated with the same host-side RK4 as the CUDA test
    (cuda_mppi.QuadrotorDynamics.step).
    """
    n_steps = int(sim_time / DT)
    t_vec = np.arange(n_steps) * DT

    # Pre-compute full reference trajectory
    ref_full = _lemniscate_ref(n_steps, t0=0.0, dt=DT)

    # Build JIT-compiled MPPI step
    mppi_step = make_mppi_step(num_samples, HORIZON, LAMBDA)

    # Host dynamics for plant propagation (same C++ RK4 as CUDA test)
    dynamics = cuda_mppi.QuadrotorDynamics()
    dynamics.mass = MASS
    dynamics.gravity = GRAVITY
    dynamics.tau_omega = TAU_OMEGA

    # Nominal control initialised at hover
    hover_thrust = MASS * GRAVITY
    u_nom = (
        jnp.zeros((HORIZON, 4), dtype=jnp.float32).at[:, 0].set(hover_thrust)
    )
    # Match the CUDA controller: the previous applied control buffer starts at
    # zero and is only overwritten after the first action is produced.
    u_prev = jnp.zeros(4, dtype=jnp.float32)
    key = jax.random.PRNGKey(0)

    # Initial state — same as test_quadrotor_tracking.py
    state = np.zeros(13, dtype=np.float32)
    state[0] = 2.0
    state[1] = 2.0
    state[2] = 0.0
    state[6] = 1.0

    positions = np.zeros((n_steps, 3), dtype=np.float32)
    controls = np.zeros((n_steps, 4), dtype=np.float32)
    comp_times = np.zeros(n_steps, dtype=np.float64)
    cost_stds: list[float] = []
    cost_means: list[float] = []
    cost_mins: list[float] = []

    print(
        f"Running JAX MPPI tracking: K={num_samples}, N={HORIZON}, "
        f"dt={DT}, λ={LAMBDA}, T_sim={sim_time}s"
    )

    for k in range(n_steps):
        positions[k] = state[:3]

        # Build sliding reference window
        ref_end = min(k + HORIZON, n_steps)
        pos_window = ref_full[k:ref_end]
        if len(pos_window) < HORIZON:
            pad = np.tile(pos_window[-1:], (HORIZON - len(pos_window), 1))
            pos_window = np.concatenate([pos_window, pad], axis=0)
        state_ref = np.zeros((HORIZON, 13), dtype=np.float32)
        state_ref[:, :3] = pos_window[:, :3]
        state_ref[:, 6] = 1.0  # identity quaternion reference

        # Shift nominal sequence by one step (drop first, repeat last)
        u_nom = jnp.concatenate([u_nom[1:], u_nom[-1:]], axis=0)

        # MPPI compute
        key, subkey = jax.random.split(key)
        x_j = jnp.array(state)
        ref_j = jnp.array(state_ref)

        t0 = time.perf_counter()
        u_nom, costs = mppi_step(subkey, x_j, u_nom, u_prev, ref_j)
        jax.block_until_ready(u_nom)
        comp_times[k] = (time.perf_counter() - t0) * 1e3

        if diagnose and k % 10 == 0:
            c = np.array(costs)
            cost_stds.append(c.std())
            cost_means.append(c.mean())
            cost_mins.append(c.min())

        action = np.array(u_nom[0])
        controls[k] = action
        u_prev = u_nom[0]

        # Propagate plant with host C++ RK4
        state = dynamics.step(state, action, DT)

        if (k + 1) % 500 == 0 or k == 0:
            pos_err = np.linalg.norm(positions[k] - ref_full[k, :3])
            print(
                f"  step {k + 1:5d}/{n_steps}  |  "
                f"pos_err={pos_err:.3f} m  |  "
                f"comp={comp_times[k]:.2f} ms"
            )

    errors = positions - ref_full[:, :3]
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
        print(f"  Current λ={LAMBDA}  →  c_std/λ = {c_std_med / LAMBDA:.2f}")
        print()
        magnitude = 10 ** int(np.log10(max(c_std_med, 1)))
        candidates = sorted(
            set(
                int(magnitude * f)
                for f in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
                if int(magnitude * f) > 0
            )
        )
        best_lam, best_score = LAMBDA, -1.0
        for lam_c in candidates:
            ratio = c_std_med / lam_c
            good = 1.0 <= ratio <= 10.0
            flag = "  ← good" if good else ""
            score = 1.0 / abs(ratio - 3.0) if good else 0.0
            if good and score > best_score:
                best_score, best_lam = score, lam_c
            print(f"  λ = {lam_c:>8}  |  c_std/λ = {ratio:6.2f}{flag}")
        print(
            f"\n  Suggested λ ≈ {best_lam}  (c_std/λ = {c_std_med / best_lam:.2f})"
        )

    if plot:
        _plot_tracking(
            t_vec, positions, ref_full[:, :3], controls, comp_times, rmse_total
        )

    return rmse_total


def run_tracking_cuda(
    num_samples: int = 900,
    sim_time: float = 30.0,
) -> tuple[float, np.ndarray]:
    """Run CUDA MPPI closed-loop on the lemniscate.

    Uses identical parameters to run_tracking() so results are directly
    comparable.  Returns (rmse_total, comp_times_ms).
    """
    import time as _time

    n_steps = int(sim_time / DT)
    t_vec = np.arange(n_steps) * DT  # noqa: F841

    ref_full = _lemniscate_ref(n_steps, t0=0.0, dt=DT)

    config = cuda_mppi.MPPIConfig(
        num_samples=num_samples,
        horizon=HORIZON,
        nx=13,
        nu=4,
        lambda_=LAMBDA,
        dt=DT,
    )
    config.set_control_sigma(SIGMA.tolist())

    dynamics = cuda_mppi.QuadrotorDynamics()
    dynamics.mass = MASS
    dynamics.gravity = GRAVITY
    dynamics.tau_omega = TAU_OMEGA

    cost = cuda_mppi.TrackingCost()
    cost.set_Q_pos(Q_POS.tolist())
    cost.set_Q_vel(Q_VEL.tolist())
    cost.Q_quat = float(Q_QUAT)
    cost.set_Q_omega(Q_OMEGA.tolist())
    cost.set_R(R_CTRL.tolist())
    cost.set_R_delta(R_DELTA.tolist())
    cost.terminal_weight = TERMINAL_WEIGHT

    controller = cuda_mppi.QuadrotorMPPI(config, dynamics, cost)
    hover_thrust = MASS * GRAVITY
    u_hover = np.array([hover_thrust, 0.0, 0.0, 0.0], dtype=np.float32)
    controller.set_nominal_control(u_hover)

    state = np.zeros(13, dtype=np.float32)
    state[0] = 2.0
    state[1] = 2.0
    state[2] = 0.0
    state[6] = 1.0

    positions = np.zeros((n_steps, 3), dtype=np.float32)
    controls = np.zeros((n_steps, 4), dtype=np.float32)
    comp_times = np.zeros(n_steps, dtype=np.float64)

    print(
        f"Running CUDA MPPI tracking: K={num_samples}, N={HORIZON}, "
        f"dt={DT}, λ={LAMBDA}, T_sim={sim_time}s"
    )

    for k in range(n_steps):
        positions[k] = state[:3]

        ref_end = min(k + HORIZON, n_steps)
        pos_window = ref_full[k:ref_end]
        if len(pos_window) < HORIZON:
            pad = np.tile(pos_window[-1:], (HORIZON - len(pos_window), 1))
            pos_window = np.concatenate([pos_window, pad], axis=0)
        state_ref = np.zeros((HORIZON, 13), dtype=np.float32)
        state_ref[:, :3] = pos_window[:, :3]
        state_ref[:, 6] = 1.0
        controller.set_state_reference(state_ref.flatten(), HORIZON)

        if k > 0:
            controller.set_applied_control(controls[k - 1])

        controller.shift()
        t0 = _time.perf_counter()
        controller.compute(state)
        comp_times[k] = (_time.perf_counter() - t0) * 1e3

        action = controller.get_action()
        controls[k] = action
        state = dynamics.step(state, action, DT)

        if (k + 1) % 500 == 0 or k == 0:
            pos_err = np.linalg.norm(positions[k] - ref_full[k, :3])
            print(
                f"  step {k + 1:5d}/{n_steps}  |  "
                f"pos_err={pos_err:.3f} m  |  "
                f"comp={comp_times[k]:.2f} ms"
            )

    errors = positions - ref_full[:, :3]
    rmse_total = float(np.sqrt(np.mean(np.sum(errors**2, axis=1))))
    return rmse_total, comp_times, positions, ref_full[:, :3]


# ---------------------------------------------------------------------------
# Head-to-head comparison
# ---------------------------------------------------------------------------


def compare(
    num_samples: int = 900,
    sim_time: float = 30.0,
    bench_samples: list[int] | None = None,
    n_iter: int = 500,
    plot: bool = False,
) -> None:
    """Run JAX and CUDA MPPI side-by-side: tracking accuracy + speed."""
    if bench_samples is None:
        bench_samples = [256, 512, 1024, 2048, 4096]

    # ── 1. Tracking accuracy ──────────────────────────────────────────────
    print("=" * 60)
    print("TRACKING ACCURACY")
    print("=" * 60)

    rmse_cuda, comp_cuda, pos_cuda, ref = run_tracking_cuda(
        num_samples=num_samples, sim_time=sim_time
    )
    print()
    rmse_jax, comp_jax, pos_jax, _ = _run_tracking_with_times(
        num_samples=num_samples, sim_time=sim_time
    )

    print()
    print("=" * 60)
    print(f"{'':20s}  {'CUDA':>10}  {'JAX':>10}")
    print("-" * 44)
    print(f"{'RMSE total (m)':20s}  {rmse_cuda:>10.4f}  {rmse_jax:>10.4f}")
    print(
        f"{'Compute mean (ms)':20s}  "
        f"{np.mean(comp_cuda):>10.2f}  {np.mean(comp_jax):>10.2f}"
    )
    print(
        f"{'Compute median (ms)':20s}  "
        f"{np.median(comp_cuda):>10.2f}  {np.median(comp_jax):>10.2f}"
    )
    print(
        f"{'Compute p95 (ms)':20s}  "
        f"{np.percentile(comp_cuda, 95):>10.2f}  {np.percentile(comp_jax, 95):>10.2f}"
    )
    print("=" * 60)

    # ── 2. Speed benchmark ────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SPEED BENCHMARK")
    print("=" * 60)
    bench(sample_counts=bench_samples, n_iter=n_iter, plot=False)

    if plot:
        _plot_compare(
            sim_time, comp_cuda, comp_jax, rmse_cuda, rmse_jax,
            pos_cuda, pos_jax, ref,
        )


def _run_tracking_with_times(
    num_samples: int = 900,
    sim_time: float = 30.0,
) -> tuple[float, np.ndarray]:
    """Like run_tracking() but returns (rmse, comp_times_ms) without plotting."""
    n_steps = int(sim_time / DT)

    ref_full = _lemniscate_ref(n_steps, t0=0.0, dt=DT)
    mppi_step = make_mppi_step(num_samples, HORIZON, LAMBDA)

    dynamics = cuda_mppi.QuadrotorDynamics()
    dynamics.mass = MASS
    dynamics.gravity = GRAVITY
    dynamics.tau_omega = TAU_OMEGA

    hover_thrust = MASS * GRAVITY
    u_nom = (
        jnp.zeros((HORIZON, 4), dtype=jnp.float32).at[:, 0].set(hover_thrust)
    )
    u_prev = jnp.zeros(4, dtype=jnp.float32)
    key = jax.random.PRNGKey(0)

    state = np.zeros(13, dtype=np.float32)
    state[0] = 2.0
    state[1] = 2.0
    state[6] = 1.0

    positions = np.zeros((n_steps, 3), dtype=np.float32)
    controls = np.zeros((n_steps, 4), dtype=np.float32)
    comp_times = np.zeros(n_steps, dtype=np.float64)

    print(
        f"Running JAX MPPI tracking: K={num_samples}, N={HORIZON}, "
        f"dt={DT}, λ={LAMBDA}, T_sim={sim_time}s"
    )

    for k in range(n_steps):
        positions[k] = state[:3]

        ref_end = min(k + HORIZON, n_steps)
        pos_window = ref_full[k:ref_end]
        if len(pos_window) < HORIZON:
            pad = np.tile(pos_window[-1:], (HORIZON - len(pos_window), 1))
            pos_window = np.concatenate([pos_window, pad], axis=0)
        state_ref = np.zeros((HORIZON, 13), dtype=np.float32)
        state_ref[:, :3] = pos_window[:, :3]
        state_ref[:, 6] = 1.0

        u_nom = jnp.concatenate([u_nom[1:], u_nom[-1:]], axis=0)

        key, subkey = jax.random.split(key)
        x_j = jnp.array(state)
        ref_j = jnp.array(state_ref)

        t0 = time.perf_counter()
        u_nom, _ = mppi_step(subkey, x_j, u_nom, u_prev, ref_j)
        jax.block_until_ready(u_nom)
        comp_times[k] = (time.perf_counter() - t0) * 1e3

        action = np.array(u_nom[0])
        controls[k] = action
        u_prev = u_nom[0]
        state = dynamics.step(state, action, DT)

        if (k + 1) % 500 == 0 or k == 0:
            pos_err = np.linalg.norm(positions[k] - ref_full[k, :3])
            print(
                f"  step {k + 1:5d}/{n_steps}  |  "
                f"pos_err={pos_err:.3f} m  |  "
                f"comp={comp_times[k]:.2f} ms"
            )

    errors = positions - ref_full[:, :3]
    rmse_total = float(np.sqrt(np.mean(np.sum(errors**2, axis=1))))
    return rmse_total, comp_times, positions, ref_full[:, :3]


def _plot_compare(
    sim_time, comp_cuda, comp_jax, rmse_cuda, rmse_jax,
    pos_cuda, pos_jax, ref,
):
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    plt.style.use(["science", "ieee"])
    plt.rcParams["figure.dpi"] = 150

    n_steps = len(comp_cuda)
    t = np.arange(n_steps) * DT

    # ── Figure 1: 3D trajectory comparison (side-by-side) ────────────────
    fig3d = plt.figure(figsize=(7.16, 3.3))
    titles = [
        f"CUDA MPPI — RMSE {rmse_cuda:.3f} m",
        f"JAX MPPI  — RMSE {rmse_jax:.3f} m",
    ]
    for col, (pos, title) in enumerate(
        zip([pos_cuda, pos_jax], titles), start=1
    ):
        ax = fig3d.add_subplot(1, 2, col, projection="3d")
        ax.plot(
            ref[:, 0], ref[:, 1], -ref[:, 2],
            color="tab:orange", lw=0.7, label="Reference",
        )
        ax.plot(
            pos[:, 0], pos[:, 1], -pos[:, 2],
            color="tab:blue", lw=0.6, label="Tracked",
        )
        ax.scatter(
            pos[0, 0], pos[0, 1], -pos[0, 2],
            c="green", s=15, zorder=5, label="Start",
        )
        ax.set_xlabel("X (m)", fontsize=6)
        ax.set_ylabel("Y (m)", fontsize=6)
        ax.set_zlabel("Z (m)", fontsize=6)
        ax.tick_params(labelsize=5)
        ax.set_title(title, fontsize=7)
        ax.legend(fontsize=5)
    fig3d.suptitle("Lemniscate Tracking — JAX vs CUDA MPPI", fontsize=8)
    fig3d.savefig("compare_trajectory_3d.png", dpi=300, bbox_inches="tight")
    print("Saved compare_trajectory_3d.png")

    # ── Figure 2: latency ─────────────────────────────────────────────────
    C_CUDA = "#E07B39"   # medium orange
    C_JAX  = "#87CEEB"   # light blue (sky blue)

    fig, ax = plt.subplots(figsize=(3.3, 2.5), constrained_layout=True)

    p99 = max(np.percentile(comp_cuda, 99), np.percentile(comp_jax, 99))
    ax.plot(t, comp_cuda, color=C_CUDA, lw=0.4, alpha=0.8, label="CUDA")
    ax.plot(t, comp_jax,  color=C_JAX,  lw=0.4, alpha=0.8, label="JAX")
    ax.axhline(20.0, color="k", ls="--", lw=0.6, label="20 ms budget")
    ax.set_ylim(0, p99)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Compute time (ms)")
    ax.set_title("Compute Latency")
    ax.legend(fontsize=6)

    fig.savefig("compare_latency.png", dpi=300, bbox_inches="tight")
    print("Saved compare_latency.png")
    plt.show()


def _plot_tracking(t, pos, ref, ctrl, comp_t, rmse):
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    plt.style.use(["science", "ieee"])
    plt.rcParams["figure.dpi"] = 150

    fig = plt.figure(figsize=(3.3, 3.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ref[:, 0], ref[:, 1], -ref[:, 2], "r--", lw=0.8, label="Reference")
    ax.plot(pos[:, 0], pos[:, 1], -pos[:, 2], "b-", lw=0.6, label="JAX MPPI")
    ax.scatter(
        *pos[0, :2], -pos[0, 2], c="green", s=20, zorder=5, label="Start"
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"JAX MPPI Lemniscate Tracking — RMSE: {rmse:.3f} m")
    ax.legend(fontsize=6)
    fig.savefig("jax_tracking_3d.png", dpi=300, bbox_inches="tight")

    labels = ["X (m)", "Y (m)", "Z (m)"]
    fig, axes = plt.subplots(3, 1, figsize=(3.3, 4.0), constrained_layout=True)
    for i, (ax, lab) in enumerate(zip(axes, labels)):
        ref_plot = ref[:, i] if i < 2 else -ref[:, i]
        pos_plot = pos[:, i] if i < 2 else -pos[:, i]
        ax.plot(t, ref_plot, "r--", lw=0.8, label="Ref")
        ax.plot(t, pos_plot, "b-", lw=0.6, label="JAX MPPI")
        ax.set_ylabel(lab)
        if i == 0:
            ax.legend(fontsize=6)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("JAX MPPI Position Tracking")
    fig.savefig("jax_tracking_axes.png", dpi=300, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(3.3, 2.0), constrained_layout=True)
    ax.plot(t, comp_t, "b-", lw=0.4, alpha=0.6)
    ax.axhline(20.0, color="r", ls="--", lw=0.8, label="20 ms budget")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Compute time (ms)")
    ax.set_title("JAX MPPI Computation Time")
    ax.legend(fontsize=6)
    fig.savefig("jax_tracking_compute.png", dpi=300, bbox_inches="tight")

    plt.show()


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _bench_cuda(
    num_samples: int, n_iter: int, x0: np.ndarray, ref: np.ndarray
) -> np.ndarray:
    """Time n_iter CUDA MPPI iterations, return array of elapsed ms."""
    config = cuda_mppi.MPPIConfig(
        num_samples=num_samples,
        horizon=HORIZON,
        nx=13,
        nu=4,
        lambda_=LAMBDA,
        dt=DT,
    )
    config.set_control_sigma(SIGMA.tolist())
    dynamics = cuda_mppi.QuadrotorDynamics()
    dynamics.mass = MASS
    dynamics.gravity = GRAVITY
    dynamics.tau_omega = TAU_OMEGA
    cost = cuda_mppi.TrackingCost()
    cost.set_Q_pos([5e3, 5e3, 5e3])
    cost.set_Q_vel([4e1, 4e1, 4e1])
    cost.Q_quat = Q_QUAT
    cost.set_Q_omega([2e1, 2e1, 2e1])
    cost.set_R([2e-2, 2e-1, 2e-1, 2e-1])
    cost.set_R_delta([1e-3, 1e-2, 1e-2, 1e-2])
    cost.terminal_weight = TERMINAL_WEIGHT

    controller = cuda_mppi.QuadrotorMPPI(config, dynamics, cost)
    hover = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0], dtype=np.float32)
    controller.set_nominal_control(hover)
    controller.set_state_reference(ref.flatten(), HORIZON)

    # warm-up
    for _ in range(5):
        controller.shift()
        controller.compute(x0)

    times = np.empty(n_iter)
    for i in range(n_iter):
        controller.shift()
        t0 = time.perf_counter()
        controller.compute(x0)
        times[i] = (time.perf_counter() - t0) * 1e3
    return times


def _bench_jax(
    num_samples: int, n_iter: int, x0: np.ndarray, ref: np.ndarray
) -> np.ndarray:
    """Time n_iter JAX MPPI iterations, return array of elapsed ms."""
    mppi_step = make_mppi_step(num_samples, HORIZON, LAMBDA)

    x0_j = jnp.array(x0)
    ref_j = jnp.array(ref)
    u_nom = jnp.zeros((HORIZON, 4), dtype=jnp.float32)
    u_nom = u_nom.at[:, 0].set(MASS * GRAVITY)  # initialise at hover thrust
    u_prev = jnp.array([MASS * GRAVITY, 0.0, 0.0, 0.0], dtype=jnp.float32)
    key = jax.random.PRNGKey(0)

    # warm-up (also triggers JIT compilation)
    for _ in range(5):
        key, subkey = jax.random.split(key)
        u_nom, _ = mppi_step(subkey, x0_j, u_nom, u_prev, ref_j)
        jax.block_until_ready(u_nom)

    times = np.empty(n_iter)
    for i in range(n_iter):
        key, subkey = jax.random.split(key)
        t0 = time.perf_counter()
        u_nom, _ = mppi_step(subkey, x0_j, u_nom, u_prev, ref_j)
        jax.block_until_ready(u_nom)
        times[i] = (time.perf_counter() - t0) * 1e3
    return times


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def bench(
    sample_counts: list[int] | None = None,
    n_iter: int = 500,
    plot: bool = False,
) -> None:
    if sample_counts is None:
        sample_counts = [256, 512, 1024, 2048, 4096]

    # Fixed state: lemniscate starting pose
    x0 = np.zeros(13, dtype=np.float32)
    x0[0] = 5.0  # px on lemniscate
    x0[2] = -2.5  # pz (NED altitude)
    x0[6] = 1.0  # qw

    ref = _lemniscate_ref(HORIZON)

    print(
        f"\nBenchmark: JAX vs CUDA MPPI  (N={HORIZON}, dt={DT}, λ={LAMBDA}, n_iter={n_iter})\n"
    )
    print(
        f"{'K':>6}  {'CUDA mean':>10}  {'CUDA std':>9}  {'JAX mean':>9}  {'JAX std':>8}  {'Speedup':>8}"
    )
    print("-" * 62)

    cuda_means, jax_means, speedups = [], [], []

    # Run all CUDA benchmarks first to avoid GPU contention with JAX
    cuda_results: dict[int, np.ndarray] = {}
    print("Running CUDA benchmarks...")
    for K in sample_counts:
        cuda_results[K] = _bench_cuda(K, n_iter, x0, ref)

    # Then all JAX benchmarks
    print("Running JAX benchmarks...")
    for K in sample_counts:
        jax_t = _bench_jax(K, n_iter, x0, ref)
        cuda_t = cuda_results[K]

        cm, cs = cuda_t.mean(), cuda_t.std()
        jm, js = jax_t.mean(), jax_t.std()
        speedup = jm / cm

        cuda_means.append(cm)
        jax_means.append(jm)
        speedups.append(speedup)

        print(
            f"{K:>6}  {cm:>9.2f}ms  {cs:>8.2f}ms  {jm:>8.2f}ms  {js:>7.2f}ms  {speedup:>7.1f}×"
        )

    print()

    if plot:
        _plot(sample_counts, cuda_means, jax_means, speedups)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot(
    sample_counts: list[int],
    cuda_ms: list[float],
    jax_ms: list[float],
    speedups: list[float],
) -> None:
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    plt.style.use(["science", "ieee"])
    plt.rcParams["figure.dpi"] = 150

    K = np.array(sample_counts)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5), constrained_layout=True)

    # --- Latency comparison ---
    ax = axes[0]
    ax.plot(K, cuda_ms, "o-", lw=0.8, ms=3, label="CUDA")
    ax.plot(K, jax_ms, "s--", lw=0.8, ms=3, label="JAX")
    ax.set_xlabel("Samples $K$")
    ax.set_ylabel("Time per iteration (ms)")
    ax.set_title("Compute Latency")
    ax.legend(fontsize=6)

    # --- Speedup ---
    ax = axes[1]
    ax.plot(K, speedups, "^-", color="tab:green", lw=0.8, ms=3)
    ax.axhline(1.0, color="k", ls="--", lw=0.6)
    ax.set_xlabel("Samples $K$")
    ax.set_ylabel("JAX / CUDA latency ratio")
    ax.set_title("CUDA Speedup over JAX")

    fig.suptitle("JAX vs CUDA MPPI — Speed Comparison")
    fig.savefig("bench_jax_mppi.png", dpi=300, bbox_inches="tight")
    print("Saved bench_jax_mppi.png")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JAX MPPI: tracking validation and speed benchmark"
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Run closed-loop lemniscate tracking (correctness check)",
    )
    parser.add_argument(
        "--samples",
        "-K",
        type=int,
        default=900,
        help="Number of MPPI samples for --track (default: 900)",
    )
    parser.add_argument(
        "--time",
        "-T",
        type=float,
        default=30.0,
        help="Simulation time in seconds for --track (default: 30)",
    )
    parser.add_argument(
        "--bench-samples",
        nargs="+",
        type=int,
        default=[256, 512, 1024, 2048, 4096],
        help="Sample counts for speed benchmark (default: 256 512 1024 2048 4096)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=500,
        help="Timed iterations per sample count for benchmark (default: 500)",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Save and show plots"
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print cost-scale statistics to guide lambda tuning, then exit",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run JAX vs CUDA head-to-head: tracking accuracy + speed benchmark",
    )
    args = parser.parse_args()

    if args.diagnose:
        diagnose_costs(num_samples=args.samples)
        raise SystemExit(0)

    if args.compare:
        compare(
            num_samples=args.samples,
            sim_time=args.time,
            bench_samples=args.bench_samples,
            n_iter=args.n_iter,
            plot=args.plot,
        )
    elif args.track:
        rmse = run_tracking(
            num_samples=args.samples,
            sim_time=args.time,
            plot=args.plot,
            diagnose=args.diagnose,
        )
        assert rmse < 2.0, f"RMSE {rmse:.3f} m exceeds 2.0 m threshold"
        print("\n✓ JAX tracking test passed.")
    else:
        bench(
            sample_counts=args.bench_samples, n_iter=args.n_iter, plot=args.plot
        )
