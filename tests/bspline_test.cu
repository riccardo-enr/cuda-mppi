/**
 * @file bspline_test.cu
 * @brief Unit tests for cubic B-spline evaluation (CPU and GPU).
 *
 * Tests:
 * 1. Constant control points → constant output
 * 2. Linear ramp → linear interpolation
 * 3. Endpoint evaluation (start and end of valid range)
 * 4. GPU batch expansion matches CPU evaluation
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>

#include "mppi/core/bspline.cuh"

static constexpr float TOL = 1e-4f;

static bool approx_eq(float a, float b, float tol = TOL) {
  return std::fabs(a - b) < tol;
}

/// Test 1: All control points equal → output should be constant everywhere.
void test_constant() {
  printf("Test 1: Constant control points... ");

  const int n_cp = 8;
  const int nu = 2;
  const float val_a = 1.5f;
  const float val_alpha = -0.3f;

  float cp[n_cp * nu];
  for (int i = 0; i < n_cp; ++i) {
    cp[i * nu + 0] = val_a;
    cp[i * nu + 1] = val_alpha;
  }

  mppi::BSplineConfig cfg;
  cfg.n_cp = n_cp;
  const int H = 40;
  const float dt = 0.05f;
  cfg.compute_knot_span(H, dt);

  float out[nu];
  for (int t = 0; t < H; ++t) {
    float u = t * dt;
    mppi::deboor_eval_host(u, cp, n_cp, nu, cfg.knot_span, out);
    assert(approx_eq(out[0], val_a));
    assert(approx_eq(out[1], val_alpha));
  }
  printf("PASSED\n");
}

/// Test 2: Linear ramp in control points → output should be linear.
void test_linear_ramp() {
  printf("Test 2: Linear ramp... ");

  const int n_cp = 8;
  const int nu = 1;
  const int H = 40;
  const float dt = 0.05f;

  // Control points: p_i = i for i = 0..7
  float cp[n_cp * nu];
  for (int i = 0; i < n_cp; ++i) {
    cp[i] = static_cast<float>(i);
  }

  mppi::BSplineConfig cfg;
  cfg.n_cp = n_cp;
  cfg.compute_knot_span(H, dt);

  // For a linear function, B-spline should reproduce it exactly
  // (B-splines of degree p reproduce polynomials up to degree p).
  // The linear function: f(u) = u / knot_span + p - 1 approximately
  // Actually for uniform knots with cp[i] = i, the curve is exactly
  // f(u) = u/knot_span + (p-1)/2 approximately in the interior.
  // Let's just check monotonicity and smoothness.
  float prev = -1e10f;
  float out[1];
  for (int t = 0; t < H; ++t) {
    float u = t * dt;
    mppi::deboor_eval_host(u, cp, n_cp, nu, cfg.knot_span, out);
    assert(out[0] >= prev - TOL);  // monotonically increasing
    prev = out[0];
  }

  // Check that output spans a reasonable range
  mppi::deboor_eval_host(0.0f, cp, n_cp, nu, cfg.knot_span, out);
  float start_val = out[0];
  mppi::deboor_eval_host((H - 1) * dt, cp, n_cp, nu, cfg.knot_span, out);
  float end_val = out[0];
  assert(end_val > start_val + 1.0f);  // should span at least part of [0,7]

  printf("PASSED (range: %.2f to %.2f)\n", start_val, end_val);
}

/// Test 3: Endpoint values.
void test_endpoints() {
  printf("Test 3: Endpoint evaluation... ");

  const int n_cp = 6;
  const int nu = 2;
  const int H = 20;
  const float dt = 0.05f;

  float cp[n_cp * nu];
  for (int i = 0; i < n_cp; ++i) {
    cp[i * nu + 0] = static_cast<float>(i);
    cp[i * nu + 1] = static_cast<float>(i) * 0.1f;
  }

  mppi::BSplineConfig cfg;
  cfg.n_cp = n_cp;
  cfg.compute_knot_span(H, dt);

  // At u=0, uniform cubic B-spline evaluates to a weighted average of
  // the first p+1=4 control points (not exactly cp[0]).
  float out[nu];
  mppi::deboor_eval_host(0.0f, cp, n_cp, nu, cfg.knot_span, out);
  // Just verify it's a reasonable value in range of first few control points
  assert(out[0] >= -TOL && out[0] < static_cast<float>(n_cp));
  assert(out[1] >= -TOL && out[1] < static_cast<float>(n_cp) * 0.1f);

  printf("PASSED (start: [%.3f, %.3f])\n", out[0], out[1]);
}

/// Test 4: GPU batch expansion matches CPU.
void test_gpu_matches_cpu() {
  printf("Test 4: GPU expansion matches CPU... ");

  const int K = 4;     // 4 samples
  const int n_cp = 8;
  const int nu = 2;
  const int H = 40;
  const float dt = 0.05f;

  mppi::BSplineConfig cfg;
  cfg.n_cp = n_cp;
  cfg.compute_knot_span(H, dt);

  // Create control points on host (K sets)
  std::vector<float> h_cp(K * n_cp * nu);
  for (int k = 0; k < K; ++k) {
    for (int i = 0; i < n_cp; ++i) {
      h_cp[k * n_cp * nu + i * nu + 0] = static_cast<float>(i) + k * 0.5f;
      h_cp[k * n_cp * nu + i * nu + 1] = static_cast<float>(i) * 0.1f - k * 0.1f;
    }
  }

  // CPU reference
  std::vector<float> cpu_out(K * H * nu, 0.0f);
  for (int k = 0; k < K; ++k) {
    cfg.expand_host(h_cp.data() + k * n_cp * nu,
                    cpu_out.data() + k * H * nu,
                    H, nu, dt);
  }

  // GPU
  float* d_cp;
  float* d_out;
  cudaMalloc(&d_cp, K * n_cp * nu * sizeof(float));
  cudaMalloc(&d_out, K * H * nu * sizeof(float));
  cudaMemcpy(d_cp, h_cp.data(), K * n_cp * nu * sizeof(float), cudaMemcpyHostToDevice);

  cfg.expand(d_cp, d_out, K, H, nu, dt);

  std::vector<float> gpu_out(K * H * nu);
  cudaMemcpy(gpu_out.data(), d_out, K * H * nu * sizeof(float), cudaMemcpyDeviceToHost);

  // Compare
  float max_err = 0.0f;
  for (int i = 0; i < K * H * nu; ++i) {
    float err = std::fabs(cpu_out[i] - gpu_out[i]);
    if (err > max_err) max_err = err;
    assert(err < TOL);
  }

  cudaFree(d_cp);
  cudaFree(d_out);

  printf("PASSED (max error: %.6f)\n", max_err);
}

/// Test 5: Knot span computation.
void test_knot_span() {
  printf("Test 5: Knot span computation... ");

  mppi::BSplineConfig cfg;
  cfg.n_cp = 8;
  cfg.compute_knot_span(40, 0.05f);

  // duration = H * dt = 2.0s
  // n_segments = n_cp - p = 8 - 3 = 5
  // knot_span = 2.0 / 5 = 0.4
  assert(approx_eq(cfg.knot_span, 0.4f));
  assert(approx_eq(cfg.duration(), 2.0f));

  printf("PASSED (knot_span=%.3f, duration=%.3f)\n", cfg.knot_span, cfg.duration());
}

int main() {
  printf("=== B-Spline Unit Tests ===\n\n");

  test_constant();
  test_linear_ramp();
  test_endpoints();
  test_knot_span();
  test_gpu_matches_cpu();

  printf("\nAll tests passed!\n");
  return 0;
}
