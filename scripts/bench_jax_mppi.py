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
LAMBDA = 1000.0
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
    R = jnp.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
        ]
    )
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
    ) -> jax.Array:
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
        return u_nom_new

    return mppi_step


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
# Benchmark helpers
# ---------------------------------------------------------------------------


def _bench_cuda(num_samples: int, n_iter: int, x0: np.ndarray, ref: np.ndarray) -> np.ndarray:
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


def _bench_jax(num_samples: int, n_iter: int, x0: np.ndarray, ref: np.ndarray) -> np.ndarray:
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
        u_nom = mppi_step(subkey, x0_j, u_nom, u_prev, ref_j)
        jax.block_until_ready(u_nom)

    times = np.empty(n_iter)
    for i in range(n_iter):
        key, subkey = jax.random.split(key)
        t0 = time.perf_counter()
        u_nom = mppi_step(subkey, x0_j, u_nom, u_prev, ref_j)
        jax.block_until_ready(u_nom)
        times[i] = (time.perf_counter() - t0) * 1e3
    return times


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def bench(
    sample_counts: list[int] | None = None,
    n_iter: int = 100,
    plot: bool = False,
) -> None:
    if sample_counts is None:
        sample_counts = [256, 512, 1024, 2048, 4096]

    # Fixed state: lemniscate starting pose
    x0 = np.zeros(13, dtype=np.float32)
    x0[0] = 5.0   # px on lemniscate
    x0[2] = -2.5  # pz (NED altitude)
    x0[6] = 1.0   # qw

    ref = _lemniscate_ref(HORIZON)

    print(f"\nBenchmark: JAX vs CUDA MPPI  (N={HORIZON}, dt={DT}, λ={LAMBDA}, n_iter={n_iter})\n")
    print(f"{'K':>6}  {'CUDA mean':>10}  {'CUDA std':>9}  {'JAX mean':>9}  {'JAX std':>8}  {'Speedup':>8}")
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
    parser = argparse.ArgumentParser(description="JAX vs CUDA MPPI benchmark")
    parser.add_argument(
        "--samples",
        nargs="+",
        type=int,
        default=[256, 512, 1024, 2048, 4096],
        help="Sample counts to benchmark",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=100,
        help="Timed iterations per sample count (default: 100)",
    )
    parser.add_argument("--plot", action="store_true", help="Save and show latency/speedup plot")
    args = parser.parse_args()

    bench(sample_counts=args.samples, n_iter=args.n_iter, plot=args.plot)
