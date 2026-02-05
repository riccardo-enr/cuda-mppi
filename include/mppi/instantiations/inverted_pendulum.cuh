#ifndef MPPI_INSTANTIATIONS_INVERTED_PENDULUM_CUH
#define MPPI_INSTANTIATIONS_INVERTED_PENDULUM_CUH

#include <cuda_runtime.h>
#include <math.h>

namespace mppi {
namespace instantiations {

struct InvertedPendulum {
    static constexpr int STATE_DIM = 4;
    static constexpr int CONTROL_DIM = 1;

    // Constants
    static constexpr float g = 9.81f;
    static constexpr float mc = 1.0f;
    static constexpr float mp = 0.1f;
    static constexpr float l = 0.5f; // Length to center of mass
    static constexpr float dt_default = 0.01f;

    __host__ __device__ void step(const float* state, const float* u, float* next_state, float dt) const {
        // State: [x, theta, x_dot, theta_dot]
        float th = state[1];
        float x_dot = state[2];
        float th_dot = state[3];
        float f = u[0];

        float sin_th = sinf(th);
        float cos_th = cosf(th);
        float total_m = mc + mp;

        // Dynamics (0 is Upright, unstable)
        // Based on standard derivation where theta=0 is up.
        // Verify: If theta is small positive, gravity should pull it further positive (unstable).
        // Common eq: th_acc = (g sin th + cos th ((-F - m l th_dot^2 sin th)/M_total)) / (l (4/3 - ...))
        // Let's stick to a known correct formulation for "0 is Up".
        
        // Cart-Pole Dynamics where theta=0 is UP:
        // x_acc = (F + mp*sin(th)*(l*th_dot^2 + g*cos(th))) / (mc + mp*sin(th)^2)
        // th_acc = (-F*cos(th) - mp*l*th_dot^2*cos(th)*sin(th) - (mc+mp)*g*sin(th)) / (l * (mc + mp*sin(th)^2))
        
        // Wait, checking signs.
        // Let's use the one from OpenAI Gym (CartPole-v1) where 0 is up.
        // temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
        // thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0/3.0 - masspole * costheta * costheta / total_mass));
        // xacc = temp - polemass_length * thetaacc * costheta / total_mass;
        
        float temp = (f + mp * l * th_dot * th_dot * sin_th) / total_m;
        float th_acc = (g * sin_th - cos_th * temp) / (l * (4.0f/3.0f - mp * cos_th * cos_th / total_m));
        float x_acc = temp - mp * l * th_acc * cos_th / total_m;

        next_state[0] = state[0] + x_dot * dt;
        next_state[1] = th + th_dot * dt;
        next_state[2] = x_dot + x_acc * dt;
        next_state[3] = th_dot + th_acc * dt;
    }
};

struct PendulumCost {
    __host__ __device__ float compute(const float* state, const float* u, int t) const {
        float x = state[0];
        float theta = state[1];
        float x_dot = state[2];
        float theta_dot = state[3];

        // Penalize distance from 0 (Upright)
        // Also penalize cart position to keep it near 0
        
        float c = 0.0f;
        c += 1.0f * x*x;
        c += 10.0f * theta*theta; // Strong penalty for angle
        c += 0.1f * x_dot*x_dot;
        c += 0.1f * theta_dot*theta_dot;
        c += 0.001f * u[0]*u[0];
        
        return c;
    }

    __host__ __device__ float terminal_cost(const float* state) const {
        float c = 0.0f;
        c += 5.0f * state[0]*state[0];
        c += 50.0f * state[1]*state[1];
        c += 1.0f * state[2]*state[2];
        c += 1.0f * state[3]*state[3];
        return c;
    }
};

} // namespace instantiations
} // namespace mppi

#endif // MPPI_INSTANTIATIONS_INVERTED_PENDULUM_CUH
