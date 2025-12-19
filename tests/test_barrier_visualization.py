#!/usr/bin/env python3
"""
Test script for barrier potential visualization tool.

This script tests the visualization functionality with a randomly initialized
actor network (no trained weights needed).
"""

import os
import sys
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from safepo.multi_agent.barrier_phs_pinn_actor import BarrierPHSPINNActor


class SimpleObsSpace:
    """Simple observation space mock for testing."""
    def __init__(self, obs_dim):
        self.shape = (obs_dim,)

class SimpleActSpace:
    """Simple action space mock for testing."""
    def __init__(self, act_dim):
        self.shape = (act_dim,)


def test_potential_computation():
    """Test barrier and task potential computation."""
    print("="*60)
    print("Testing Barrier Potential Computation")
    print("="*60)
    
    # Create simple space objects (no environment needed)
    obs_dim = 152  # Point MultiGoal observation dimension
    act_dim = 2    # Point action dimension
    
    obs_space = SimpleObsSpace(obs_dim)
    act_space = SimpleActSpace(act_dim)
    
    # Configuration
    config = {
        'device': 'cpu',
        'hidden_size': 256,
        'physics_hidden': 128,
        'pinn_state_dim': 4,
        'std_x_coef': 1.0,
        'std_y_coef': 0.5,
        'barrier_r_safe': 0.5,
        'barrier_epsilon': 0.005,
        'barrier_clip_max': 100.0,
        'num_lidar_bins': 16,
        'barrier_k_scale': 2.0,
        'barrier_gradient_scale': 1.5,
        'barrier_decay_rate': 2.0,
        'min_barrier_k': 0.5,
        'cost_aware_weight': 0.3,
        'danger_zone_threshold': 0.8,
    }
    
    # Create actor
    actor = BarrierPHSPINNActor(
        config=config,
        obs_space=obs_space,
        act_space=act_space,
        device=torch.device('cpu')
    )
    
    print(f"✓ Created actor network")
    print(f"  - Observation dim: {obs_space.shape[0]}")
    print(f"  - Action dim: {act_space.shape[0]}")
    
    # Test with dummy observation
    obs = torch.randn(4, obs_space.shape[0])  # Batch of 4
    
    # Compute barrier potential
    with torch.no_grad():
        H_barrier, grad_H_barrier = actor._compute_barrier_potential(obs)
        H_task, grad_H_task = actor._compute_task_potential(obs)
        
        print(f"\n✓ Computed potentials:")
        print(f"  - H_barrier shape: {H_barrier.shape}, range: [{H_barrier.min():.2f}, {H_barrier.max():.2f}]")
        print(f"  - H_task shape: {H_task.shape}, range: [{H_task.min():.2f}, {H_task.max():.2f}]")
        print(f"  - grad_H_barrier shape: {grad_H_barrier.shape}")
        print(f"  - grad_H_task shape: {grad_H_task.shape}")
        
        # Check shapes
        assert H_barrier.shape == (4, 1), "Barrier potential shape mismatch"
        assert H_task.shape == (4, 1), "Task potential shape mismatch"
        assert grad_H_barrier.shape == (4, 2), "Barrier gradient shape mismatch"
        assert grad_H_task.shape == (4, 2), "Task gradient shape mismatch"
        
        # Check values are reasonable
        assert torch.all(H_barrier >= 0), "Barrier potential should be non-negative"
        assert torch.all(torch.isfinite(H_barrier)), "Barrier potential contains NaN/Inf"
        assert torch.all(torch.isfinite(H_task)), "Task potential contains NaN/Inf"
        
    print(f"\n✓ All tests passed!")
    print("="*60)


def test_visualization_grid_computation():
    """Test grid-based potential computation for visualization."""
    print("\n" + "="*60)
    print("Testing Visualization Grid Computation")
    print("="*60)
    
    from safepo.utils.visualize_barrier_potential import BarrierPotentialVisualizer
    
    # Create a mock visualizer (without loading weights)
    # We'll test the grid computation logic
    
    print("Creating mock actor...")
    
    obs_dim = 152  # Point MultiGoal
    act_dim = 2
    
    obs_space = SimpleObsSpace(obs_dim)
    act_space = SimpleActSpace(act_dim)
    
    config = {
        'device': 'cpu',
        'hidden_size': 128,  # Smaller for speed
        'physics_hidden': 64,
        'pinn_state_dim': 4,
        'std_x_coef': 1.0,
        'std_y_coef': 0.5,
        'barrier_r_safe': 0.5,
        'barrier_epsilon': 0.005,
        'barrier_clip_max': 100.0,
        'num_lidar_bins': 16,
        'barrier_k_scale': 2.0,
        'barrier_gradient_scale': 1.5,
        'barrier_decay_rate': 2.0,
        'min_barrier_k': 0.5,
        'cost_aware_weight': 0.3,
        'danger_zone_threshold': 0.8,
    }
    
    actor = BarrierPHSPINNActor(
        config=config,
        obs_space=obs_space,
        act_space=act_space,
        device=torch.device('cpu')
    )
    
    # Test dummy observation creation
    class MockVisualizer:
        def __init__(self, actor, device):
            self.actor = actor
            self.device = device
            self.x_min, self.x_max = -2.5, 2.5
            self.y_min, self.y_max = -2.5, 2.5
        
        def _create_dummy_observation(self, x, y, vx=0.0, vy=0.0):
            obs_dim = self.actor.obs_dim
            obs = np.zeros(obs_dim, dtype=np.float32)
            obs[3] = vx
            obs[4] = vy
            obs[12:] = 0.1
            
            dist_to_boundary = min(
                abs(x - self.x_min),
                abs(x - self.x_max),
                abs(y - self.y_min),
                abs(y - self.y_max)
            )
            
            if dist_to_boundary < 0.5:
                boundary_lidar = np.exp(-dist_to_boundary * 2)
                obs[44:60] = boundary_lidar
            
            return torch.from_numpy(obs).unsqueeze(0).to(self.device)
    
    mock_viz = MockVisualizer(actor, torch.device('cpu'))
    
    # Test grid computation (small grid for speed)
    resolution = 5
    print(f"\nComputing {resolution}x{resolution} grid...")
    
    x = np.linspace(-2.5, 2.5, resolution)
    y = np.linspace(-2.5, 2.5, resolution)
    X, Y = np.meshgrid(x, y)
    
    H_barrier_grid = np.zeros_like(X)
    H_task_grid = np.zeros_like(X)
    
    with torch.no_grad():
        for i in range(resolution):
            for j in range(resolution):
                obs = mock_viz._create_dummy_observation(X[i, j], Y[i, j])
                H_b, _ = actor._compute_barrier_potential(obs)
                H_t, _ = actor._compute_task_potential(obs)
                
                H_barrier_grid[i, j] = H_b.item()
                H_task_grid[i, j] = H_t.item()
    
    print(f"✓ Grid computation complete")
    print(f"  - Barrier potential range: [{H_barrier_grid.min():.2f}, {H_barrier_grid.max():.2f}]")
    print(f"  - Task potential range: [{H_task_grid.min():.2f}, {H_task_grid.max():.2f}]")
    
    # Check grid values
    assert np.all(np.isfinite(H_barrier_grid)), "Grid contains NaN/Inf in barrier potential"
    assert np.all(np.isfinite(H_task_grid)), "Grid contains NaN/Inf in task potential"
    
    print(f"\n✓ Grid computation tests passed!")
    print("="*60)


if __name__ == '__main__':
    print("\n" + "🔬 Barrier Potential Visualization Tests 🔬".center(60))
    print()
    
    try:
        test_potential_computation()
        test_visualization_grid_computation()
        
        print("\n" + "="*60)
        print("✅ All tests passed successfully!".center(60))
        print("="*60)
        print("\nNext steps:")
        print("1. Train a MAPPO-Safe-PINN model:")
        print("   cd safepo/multi_agent")
        print("   python mappo_safe_pinn.py --task SafetyPointMultiGoal1-v0")
        print()
        print("2. Visualize the trained model:")
        print("   ./scripts/visualize_potential.sh -m runs/multi_goal/models_seed0")
        print()
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ Test failed!".center(60))
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
