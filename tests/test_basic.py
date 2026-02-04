import pytest
import numpy as np
import cuda_mppi

def test_config_instantiation():
    config = cuda_mppi.MPPIConfig(
        num_samples=100,
        horizon=20,
        nx=4,
        nu=2,
        lambda_=1.0,
        dt=0.1,
        u_scale=1.0,
        w_action_seq_cost=0.0,
        num_support_pts=0
    )
    assert config.num_samples == 100
    assert config.horizon == 20
    assert config.nx == 4

def test_double_integrator_mppi():
    config = cuda_mppi.MPPIConfig(
        num_samples=128,
        horizon=20,
        nx=4,
        nu=2,
        lambda_=1.0,
        dt=0.05,
        u_scale=1.0,
        w_action_seq_cost=0.0,
        num_support_pts=0
    )
    
    # Just checking instantiation and basic run
    # Dynamics and Cost are default instantiated in C++ binding
    controller = cuda_mppi.DoubleIntegratorMPPI(config)
    
    state = np.zeros(4, dtype=np.float32)
    # The compute method expects float array
    controller.compute(state)
    
    action = controller.get_action()
    assert action.shape == (2,)
    assert isinstance(action, np.ndarray)
