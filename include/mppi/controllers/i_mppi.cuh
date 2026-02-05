#ifndef IMPPI_CONTROLLER_CUH
#define IMPPI_CONTROLLER_CUH

#include "mppi/controllers/mppi.cuh"

namespace mppi {

__global__ void apply_bias_kernel(
    float* noise,
    const float* u_nom,
    const float* u_ref,
    int num_samples,
    int horizon,
    int nu,
    int start_biased_idx
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_samples) return;
    
    // Only apply bias to samples >= start_biased_idx
    if (k < start_biased_idx) return;

    // Shift: noise = noise + (u_ref - u_nom)
    // Effectively making u = u_nom + noise_shifted = u_ref + noise_original
    
    for(int t=0; t<horizon; ++t) {
        for(int i=0; i<nu; ++i) {
            int idx = k * (horizon * nu) + t * nu + i;
            int u_idx = t * nu + i;
            
            float shift = u_ref[u_idx] - u_nom[u_idx];
            noise[idx] += shift;
        }
    }
}

template <typename Dynamics, typename Cost>
class IMPPIController : public MPPIController<Dynamics, Cost> {
public:
    IMPPIController(const MPPIConfig& config, const Dynamics& dynamics, const Cost& cost)
        : MPPIController<Dynamics, Cost>(config, dynamics, cost) {
        
        // Allocate memory for reference trajectory
        HANDLE_ERROR(cudaMalloc(&d_u_ref_, config.horizon * config.nu * sizeof(float)));
        HANDLE_ERROR(cudaMemset(d_u_ref_, 0, config.horizon * config.nu * sizeof(float)));
    }

    ~IMPPIController() {
        cudaFree(d_u_ref_);
    }

    void set_reference_trajectory(const Eigen::VectorXf& u_ref_flat) {
        // u_ref_flat should be size T * nu
        if (u_ref_flat.size() != this->config_.horizon * this->config_.nu) {
            std::cerr << "Error: Reference trajectory size mismatch!" << std::endl;
            return;
        }
        HANDLE_ERROR(cudaMemcpy(d_u_ref_, u_ref_flat.data(), 
                                this->config_.horizon * this->config_.nu * sizeof(float), 
                                cudaMemcpyHostToDevice));
    }

    // Override compute to include biased sampling
    void compute(const Eigen::VectorXf& state) {
        // 1. Copy state to device
        HANDLE_ERROR(cudaMemcpy(this->d_initial_state_, state.data(), 
                                this->config_.nx * sizeof(float), cudaMemcpyHostToDevice));

        // 2. Sample Standard Noise (Normal Distribution)
        HANDLE_CURAND_ERROR(curandGenerateNormal(this->gen_, this->d_noise_, 
                                                 this->config_.num_samples * this->config_.horizon * this->config_.nu, 
                                                 0.0f, 1.0f));

        // 3. Apply Bias to a subset of samples
        // alpha determines fraction of samples to be biased towards u_ref
        int num_biased = (int)(this->config_.num_samples * this->config_.alpha);
        int start_biased_idx = this->config_.num_samples - num_biased;
        
        if (num_biased > 0) {
            dim3 block(256);
            dim3 grid((this->config_.num_samples + block.x - 1) / block.x);
            
            apply_bias_kernel<<<grid, block>>>(
                this->d_noise_,
                this->d_u_nom_,
                d_u_ref_,
                this->config_.num_samples,
                this->config_.horizon,
                this->config_.nu,
                start_biased_idx
            );
            HANDLE_ERROR(cudaGetLastError());
        }

        // 4. Launch Rollout Kernel (Standard)
        // The noise is now "shifted" for biased samples, so standard rollout
        // u = u_nom + noise will produce trajectories centered at u_ref for those samples.
        dim3 block(256);
        dim3 grid((this->config_.num_samples + block.x - 1) / block.x);
        
        kernels::rollout_kernel<<<grid, block>>>(
            this->dynamics_,
            this->cost_,
            this->config_,
            this->d_initial_state_,
            this->d_u_nom_,
            this->d_noise_,
            this->d_costs_
        );
        HANDLE_ERROR(cudaGetLastError());
        HANDLE_ERROR(cudaDeviceSynchronize());

        // 5. Compute Weights (Standard Softmax)
        // Note: For biased importance sampling, technically we need to adjust weights 
        // by the ratio q(u)/p(u).
        // However, I-MPPI usually employs a mixture distribution approach where 
        // samples are treated equally in the softmax if they are drawn from the mixture.
        // The standard MPPI weight formula w ~ exp(-cost/lambda) works for the mixture Q
        // if we consider Q as the sampling distribution.
        // Let's stick to standard weighting for now as per "Biased-MPPI" common impl.
        
        // Simple Host-side Weighting (copied from base for prototype)
        std::vector<float> h_costs(this->config_.num_samples);
        HANDLE_ERROR(cudaMemcpy(h_costs.data(), this->d_costs_, 
                                this->config_.num_samples * sizeof(float), cudaMemcpyDeviceToHost));

        float min_cost = h_costs[0];
        for(float c : h_costs) if(c < min_cost) min_cost = c;

        std::vector<float> h_weights(this->config_.num_samples);
        float sum_weights = 0.0f;
        for(int k=0; k<this->config_.num_samples; ++k) {
            float w = expf(-(h_costs[k] - min_cost) / this->config_.lambda);
            h_weights[k] = w;
            sum_weights += w;
        }

        for(int k=0; k<this->config_.num_samples; ++k) {
            h_weights[k] /= sum_weights;
        }

        // 6. Update U_nom
        HANDLE_ERROR(cudaMemcpy(this->d_weights_, h_weights.data(), 
                                this->config_.num_samples * sizeof(float), cudaMemcpyHostToDevice));
        
        int num_params = this->config_.horizon * this->config_.nu;
        int threads = 256;
        int blocks = (num_params + threads - 1) / threads;
        
        weighted_update_kernel<<<blocks, threads>>>(
            this->d_u_nom_,
            this->d_noise_,
            this->d_weights_,
            this->config_.num_samples,
            num_params,
            0.1f 
        );
        HANDLE_ERROR(cudaGetLastError());
    }

private:
    float* d_u_ref_;
};

} // namespace mppi

#endif // IMPPI_CONTROLLER_CUH
