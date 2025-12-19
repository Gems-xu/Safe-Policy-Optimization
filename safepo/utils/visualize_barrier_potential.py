#!/usr/bin/env python3
# Copyright 2025 Gems Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Barrier Potential Visualization Tool for Safe-pH-MARL

This script visualizes the learned barrier potential field and task potential field
from the Barrier Port-Hamiltonian PINN Actor in Multi-Goal environments.

Usage:
    python visualize_barrier_potential.py \
        --model_dir runs/multi_goal/models_seed0 \
        --task SafetyPointMultiGoal1-v0 \
        --agent_id 0 \
        --output_dir visualization_output

Features:
    1. 2D Heatmap of barrier potential H_barrier(x, y)
    2. 2D Heatmap of task potential H_task(x, y)
    3. Combined total potential H_total(x, y)
    4. Gradient vector field showing force directions
    5. Sample trajectories overlaid on potential fields
    6. Interactive plots with obstacle/goal positions
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Tuple

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from safepo.multi_agent.barrier_phs_pinn_actor import BarrierPHSPINNActor


class SimpleObsSpace:
    """Simple observation space mock for visualization without environment."""
    def __init__(self, obs_dim):
        self.shape = (obs_dim,)

class SimpleActSpace:
    """Simple action space mock for visualization without environment."""
    def __init__(self, act_dim):
        self.shape = (act_dim,)


# Environment-specific observation dimensions
TASK_OBS_DIMS = {
    'SafetyPointMultiGoal1-v0': 152,
    'SafetyPointMultiGoal2-v0': 168,
    'SafetyCarMultiGoal1-v0': 176,  # Car has more sensors than Point
    'SafetyCarMultiGoal2-v0': 192,
    'SafetyAntMultiGoal1-v0': 196,
    'SafetyAntMultiGoal2-v0': 212,
}

# Action dimensions
TASK_ACT_DIMS = {
    'SafetyPointMultiGoal1-v0': 2,
    'SafetyPointMultiGoal2-v0': 2,
    'SafetyCarMultiGoal1-v0': 2,
    'SafetyCarMultiGoal2-v0': 2,
    'SafetyAntMultiGoal1-v0': 8,
    'SafetyAntMultiGoal2-v0': 8,
}


class BarrierPotentialVisualizer:
    """
    Visualize learned barrier potential and task potential fields.
    """
    
    def __init__(self, model_dir, task, agent_id=0, device='cpu'):
        """
        Initialize visualizer.
        
        Args:
            model_dir: Directory containing trained model checkpoints
            task: Environment task name (e.g., 'SafetyPointMultiGoal1-v0')
            agent_id: Which agent's policy to visualize
            device: Torch device
        """
        self.model_dir = model_dir
        self.task = task
        self.agent_id = agent_id
        self.device = torch.device(device)
        
        # Get observation and action dimensions from task name
        if task not in TASK_OBS_DIMS:
            print(f"Warning: Unknown task '{task}', using default dimensions")
            print(f"Available tasks: {list(TASK_OBS_DIMS.keys())}")
            obs_dim = 152  # Default for Point
            act_dim = 2
        else:
            obs_dim = TASK_OBS_DIMS[task]
            act_dim = TASK_ACT_DIMS[task]
        
        # Create simple space objects (no environment needed)
        obs_space = SimpleObsSpace(obs_dim)
        act_space = SimpleActSpace(act_dim)
        
        # Load model configuration (try to infer from checkpoint)
        model_config = self._load_model_config()
        
        # Create actor network
        self.actor = BarrierPHSPINNActor(
            config=model_config,
            obs_space=obs_space,
            act_space=act_space,
            device=self.device
        )
        
        # Load trained weights
        self._load_weights()
        self.actor.eval()
        
        # Environment bounds (for Multi-Goal, typically [-2, 2] for x and y)
        self.x_min, self.x_max = -2.5, 2.5
        self.y_min, self.y_max = -2.5, 2.5
    
    def _load_model_config(self):
        """Load model configuration from checkpoint directory."""
        # Default configuration matching mappo_safe_pinn.py
        config = {
            'device': str(self.device),
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
        return config
    
    def _load_weights(self):
        """Load trained model weights."""
        model_path = os.path.join(self.model_dir, f'actor_agent{self.agent_id}.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        state_dict = torch.load(model_path, map_location=self.device)
        self.actor.load_state_dict(state_dict)
    
    def _create_dummy_observation(self, x, y, vx=0.0, vy=0.0):
        """
        Create a dummy observation for a given position and velocity.
        
        For Multi-Goal environments, we simulate lidar readings based on position
        to create spatially-varying observations that the network can use to
        learn position-dependent potentials.
        
        Observation structure (Multi-Goal):
        - obs[0:3]: accelerometer (ax, ay, az)
        - obs[3:6]: velocimeter (vx, vy, vz)
        - obs[6:9]: gyro
        - obs[9:12]: magnetometer  
        - obs[12:28]: goal_red lidar (16 bins) - agent 0's goal
        - obs[28:44]: goal_blue lidar (16 bins) - agent 1's goal
        - obs[44:60]: hazard lidar (16 bins) - obstacles/boundaries
        - obs[60:76]: vase lidar (16 bins) - additional obstacles
        - ... (remaining components)
        
        Args:
            x, y: Position coordinates in workspace
            vx, vy: Velocity components
            
        Returns:
            obs: [1, obs_dim] observation tensor
        """
        obs_dim = self.actor.obs_dim
        obs = np.zeros(obs_dim, dtype=np.float32)
        
        # Set velocity
        obs[3] = vx
        obs[4] = vy
        
        # === Simulate Goal Lidar Readings ===
        # For Multi-Goal, assume two goals at fixed positions
        # Goal Red (agent 0): positioned at (1.5, 1.5) for example
        # Goal Blue (agent 1): positioned at (-1.5, -1.5) for example
        goal_red_pos = np.array([1.5, 1.5])
        goal_blue_pos = np.array([-1.5, -1.5])
        agent_pos = np.array([x, y])
        
        # Compute goal lidar readings (16 bins covering 360 degrees)
        num_lidar_bins = 16
        angles = np.linspace(0, 2*np.pi, num_lidar_bins, endpoint=False)
        
        # Goal Red lidar (obs[12:28])
        goal_red_lidar = self._compute_goal_lidar(agent_pos, goal_red_pos, angles, max_dist=3.0)
        obs[12:28] = goal_red_lidar
        
        # Goal Blue lidar (obs[28:44])
        goal_blue_lidar = self._compute_goal_lidar(agent_pos, goal_blue_pos, angles, max_dist=3.0)
        obs[28:44] = goal_blue_lidar
        
        # === Simulate Hazard/Boundary Lidar ===
        # Detect distance to boundaries in each direction
        boundary_lidar = np.zeros(num_lidar_bins, dtype=np.float32)
        for i, angle in enumerate(angles):
            # Cast a ray in this direction and find distance to boundary
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            # Find intersection with boundaries [-2.5, 2.5]
            t_min = float('inf')
            
            # Check x boundaries
            if abs(dx) > 1e-6:
                t_x_min = (self.x_min - x) / dx if dx < 0 else (self.x_max - x) / dx
                if t_x_min > 0:
                    t_min = min(t_min, t_x_min)
            
            # Check y boundaries  
            if abs(dy) > 1e-6:
                t_y_min = (self.y_min - y) / dy if dy < 0 else (self.y_max - y) / dy
                if t_y_min > 0:
                    t_min = min(t_min, t_y_min)
            
            # Convert distance to lidar reading (closer = higher)
            if t_min < float('inf'):
                dist = t_min
                # Exponential decay: reading = exp(-gain * dist)
                boundary_lidar[i] = np.exp(-0.5 * dist)
            else:
                boundary_lidar[i] = 0.0
        
        obs[44:60] = boundary_lidar  # Hazard lidar
        
        # Vase lidar: set to low values (no vases in this simplified visualization)
        if obs_dim > 76:
            obs[60:76] = 0.05
        
        # Convert to tensor
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        return obs_tensor
    
    def _compute_goal_lidar(self, agent_pos, goal_pos, angles, max_dist=3.0):
        """
        Compute lidar readings for a goal at a specific position.
        
        Args:
            agent_pos: [x, y] agent position
            goal_pos: [x, y] goal position
            angles: Array of lidar angles (radians)
            max_dist: Maximum detection distance
            
        Returns:
            lidar_readings: Array of lidar values [0, 1]
        """
        num_bins = len(angles)
        lidar = np.zeros(num_bins, dtype=np.float32)
        
        # Vector from agent to goal
        to_goal = goal_pos - agent_pos
        dist_to_goal = np.linalg.norm(to_goal)
        
        if dist_to_goal < 1e-6:
            # Agent is at goal, return high readings everywhere
            return np.ones(num_bins, dtype=np.float32) * 0.9
        
        # Angle to goal
        angle_to_goal = np.arctan2(to_goal[1], to_goal[0])
        
        # For each lidar bin, compute reading based on angle difference
        for i, angle in enumerate(angles):
            # Angular difference (wrapped to [-pi, pi])
            angle_diff = angle - angle_to_goal
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            
            # Lidar reading is high when pointing toward goal
            # Use a Gaussian-like kernel centered on the goal direction
            angular_width = np.pi / 8  # Width of lidar beam
            angular_response = np.exp(-0.5 * (angle_diff / angular_width) ** 2)
            
            # Distance-based decay
            if dist_to_goal < max_dist:
                dist_response = (max_dist - dist_to_goal) / max_dist
            else:
                dist_response = 0.0
            
            # Combined response
            lidar[i] = angular_response * dist_response
        
        return lidar
    
    @torch.no_grad()
    def compute_potential_grid(self, resolution=50, velocity=(0.0, 0.0)):
        """
        Compute barrier and task potentials over a 2D grid.
        
        Args:
            resolution: Grid resolution (number of points per axis)
            velocity: (vx, vy) velocity to use for all grid points
            
        Returns:
            X, Y: Meshgrid coordinates
            H_barrier: Barrier potential values
            H_task: Task potential values
            H_total: Total potential values
        """
        # Create grid
        x = np.linspace(self.x_min, self.x_max, resolution)
        y = np.linspace(self.y_min, self.y_max, resolution)
        X, Y = np.meshgrid(x, y)
        
        H_barrier = np.zeros_like(X)
        H_task = np.zeros_like(X)
        H_total = np.zeros_like(X)
        
        vx, vy = velocity
        
        # Compute potentials for each grid point
        for i in range(resolution):
            for j in range(resolution):
                obs = self._create_dummy_observation(X[i, j], Y[i, j], vx, vy)
                
                # Compute barrier potential
                H_b, _ = self.actor._compute_barrier_potential(obs)
                H_barrier[i, j] = H_b.item()
                
                # Compute task potential
                H_t, _ = self.actor._compute_task_potential(obs)
                H_task[i, j] = H_t.item()
                
                H_total[i, j] = H_b.item() + H_t.item()
        
        return X, Y, H_barrier, H_task, H_total
    
    @torch.no_grad()
    def compute_gradient_field(self, resolution=20, velocity=(0.0, 0.0)):
        """
        Compute gradient vector field over a 2D grid.
        
        Args:
            resolution: Grid resolution (lower than potential grid for clarity)
            velocity: (vx, vy) velocity to use for all grid points
            
        Returns:
            X, Y: Meshgrid coordinates
            grad_X, grad_Y: Gradient components (force directions)
        """
        x = np.linspace(self.x_min, self.x_max, resolution)
        y = np.linspace(self.y_min, self.y_max, resolution)
        X, Y = np.meshgrid(x, y)
        
        grad_X = np.zeros_like(X)
        grad_Y = np.zeros_like(X)
        
        vx, vy = velocity
        
        for i in range(resolution):
            for j in range(resolution):
                obs = self._create_dummy_observation(X[i, j], Y[i, j], vx, vy)
                state = self.actor._extract_physics_state(obs)
                
                # Compute total Hamiltonian gradient
                H_task, H_barrier, grad_H = self.actor._compute_total_hamiltonian_gradient(obs, state)
                
                # Port-Hamiltonian dynamics: ẋ = (J - R) ∇H
                # For visualization, we show the direction of motion (repulsive force)
                # Gradient points uphill, force points downhill
                grad_X[i, j] = -grad_H[0, 0].item()  # Negative for downhill
                grad_Y[i, j] = -grad_H[0, 1].item()
        
        return X, Y, grad_X, grad_Y
    
    def visualize_potential_2d(self, resolution=100, save_path=None, verbose=False):
        """
        Create 2D heatmap visualizations of barrier, task, and total potentials.
        
        Args:
            resolution: Grid resolution
            save_path: Directory to save figures (if None, only display)
            verbose: Whether to print progress messages
        """
        if verbose:
            print(f"Computing potential fields at {resolution}x{resolution} resolution...")
        X, Y, H_barrier, H_task, H_total = self.compute_potential_grid(resolution)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Barrier potential
        im1 = axes[0].contourf(X, Y, H_barrier, levels=20, cmap='hot')
        axes[0].set_title('Barrier Potential $H_{barrier}(x, y)$', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X Position (m)', fontsize=12)
        axes[0].set_ylabel('Y Position (m)', fontsize=12)
        fig.colorbar(im1, ax=axes[0], label='Potential Energy')
        axes[0].grid(True, alpha=0.3)
        
        # Task potential
        im2 = axes[1].contourf(X, Y, H_task, levels=20, cmap='viridis')
        axes[1].set_title('Task Potential $H_{task}(x, y)$', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('X Position (m)', fontsize=12)
        axes[1].set_ylabel('Y Position (m)', fontsize=12)
        fig.colorbar(im2, ax=axes[1], label='Potential Energy')
        axes[1].grid(True, alpha=0.3)
        
        # Total potential
        im3 = axes[2].contourf(X, Y, H_total, levels=20, cmap='coolwarm')
        axes[2].set_title('Total Potential $H_{total}(x, y)$', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('X Position (m)', fontsize=12)
        axes[2].set_ylabel('Y Position (m)', fontsize=12)
        fig.colorbar(im3, ax=axes[2], label='Potential Energy')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filepath = os.path.join(save_path, 'potential_fields_2d.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"✓ Saved 2D potential fields to {filepath}")
        
        plt.show()
    
    def visualize_potential_3d(self, resolution=50, potential_type='barrier', save_path=None, verbose=False):
        """
        Create 3D surface plot of potential field.
        
        Args:
            resolution: Grid resolution
            potential_type: 'barrier', 'task', or 'total'
            save_path: Directory to save figures
            verbose: Whether to print progress messages
        """
        if verbose:
            print(f"Computing {potential_type} potential for 3D visualization...")
        X, Y, H_barrier, H_task, H_total = self.compute_potential_grid(resolution)
        
        # Select potential to visualize
        if potential_type == 'barrier':
            Z = H_barrier
            title = 'Barrier Potential Surface $H_{barrier}(x, y)$'
            cmap = 'hot'
        elif potential_type == 'task':
            Z = H_task
            title = 'Task Potential Surface $H_{task}(x, y)$'
            cmap = 'viridis'
        else:
            Z = H_total
            title = 'Total Potential Surface $H_{total}(x, y)$'
            cmap = 'coolwarm'
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.9, 
                               linewidth=0, antialiased=True)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('X Position (m)', fontsize=12, labelpad=10)
        ax.set_ylabel('Y Position (m)', fontsize=12, labelpad=10)
        ax.set_zlabel('Potential Energy', fontsize=12, labelpad=10)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Adjust viewing angle
        ax.view_init(elev=30, azim=45)
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filepath = os.path.join(save_path, f'potential_surface_3d_{potential_type}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"✓ Saved 3D {potential_type} potential surface to {filepath}")
        
        plt.show()
    
    def visualize_gradient_field(self, resolution=20, save_path=None, verbose=False):
        """
        Visualize gradient vector field showing force directions.
        
        Args:
            resolution: Grid resolution for vector field
            save_path: Directory to save figure
            verbose: Whether to print progress messages
        """
        if verbose:
            print(f"Computing gradient vector field at {resolution}x{resolution} resolution...")
        X_grad, Y_grad, grad_X, grad_Y = self.compute_gradient_field(resolution)
        
        # Also compute barrier potential for background
        X_bg, Y_bg, H_barrier, _, _ = self.compute_potential_grid(resolution=50)
        
        fig, ax = plt.subplots(figsize=(10, 9))
        
        # Background: barrier potential heatmap
        im = ax.contourf(X_bg, Y_bg, H_barrier, levels=15, cmap='hot', alpha=0.6)
        fig.colorbar(im, ax=ax, label='Barrier Potential')
        
        # Overlay: gradient vector field
        # Normalize vectors for better visualization
        magnitude = np.sqrt(grad_X**2 + grad_Y**2) + 1e-6
        grad_X_norm = grad_X / magnitude
        grad_Y_norm = grad_Y / magnitude
        
        ax.quiver(X_grad, Y_grad, grad_X_norm, grad_Y_norm, 
                  magnitude, cmap='viridis', scale=30, width=0.003,
                  headwidth=4, headlength=5, alpha=0.8)
        
        ax.set_title('Gradient Vector Field: Force Directions\n(Background: Barrier Potential)', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filepath = os.path.join(save_path, 'gradient_vector_field.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"✓ Saved gradient vector field to {filepath}")
        
        plt.show()
    
    def visualize_slice_comparison(self, y_slice=0.0, resolution=200, save_path=None, verbose=False):
        """
        Visualize 1D slice of potentials at fixed y-coordinate.
        Useful for examining barrier sharpness and task potential shape.
        
        Args:
            y_slice: Y-coordinate to slice at
            resolution: Number of points along x-axis
            save_path: Directory to save figure
            verbose: Whether to print progress messages
        """
        if verbose:
            print(f"Computing 1D slice at y={y_slice}...")
        x = np.linspace(self.x_min, self.x_max, resolution)
        
        H_barrier_slice = []
        H_task_slice = []
        H_total_slice = []
        
        for x_val in x:
            obs = self._create_dummy_observation(x_val, y_slice)
            
            H_b, _ = self.actor._compute_barrier_potential(obs)
            H_t, _ = self.actor._compute_task_potential(obs)
            
            H_barrier_slice.append(H_b.item())
            H_task_slice.append(H_t.item())
            H_total_slice.append(H_b.item() + H_t.item())
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(x, H_barrier_slice, label='$H_{barrier}$', linewidth=2, color='red')
        ax.plot(x, H_task_slice, label='$H_{task}$', linewidth=2, color='blue')
        ax.plot(x, H_total_slice, label='$H_{total}$', linewidth=2, color='purple', linestyle='--')
        
        ax.set_title(f'Potential Energy Profile (slice at y={y_slice})', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Potential Energy', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filepath = os.path.join(save_path, f'potential_slice_y{y_slice}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"✓ Saved 1D potential slice to {filepath}")
        
        plt.show()
    
    def visualize_all(self, output_dir='vizs', verbose=False):
        """
        Generate all visualization plots and save to output directory.
        
        Args:
            output_dir: Directory to save all visualization outputs (default: vizs/)
            verbose: Whether to print progress messages (default: False)
        """
        if verbose:
            print("=" * 60)
            print("Barrier Potential Visualization Suite")
            print("=" * 60)
        
        # 2D heatmaps
        self.visualize_potential_2d(resolution=100, save_path=output_dir, verbose=verbose)
        
        # 3D surfaces
        self.visualize_potential_3d(resolution=50, potential_type='barrier', save_path=output_dir, verbose=verbose)
        self.visualize_potential_3d(resolution=50, potential_type='task', save_path=output_dir, verbose=verbose)
        self.visualize_potential_3d(resolution=50, potential_type='total', save_path=output_dir, verbose=verbose)
        
        # Gradient vector field
        self.visualize_gradient_field(resolution=20, save_path=output_dir, verbose=verbose)
        
        # 1D slices
        self.visualize_slice_comparison(y_slice=0.0, save_path=output_dir, verbose=verbose)
        
        if verbose:
            print("=" * 60)
            print(f"✓ All visualizations saved to {output_dir}")
            print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize learned barrier potential from Barrier-PHS-PINN Actor'
    )
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained model checkpoints')
    parser.add_argument('--task', type=str, default='SafetyPointMultiGoal1-v0',
                        help='Environment task name')
    parser.add_argument('--agent_id', type=int, default=0,
                        help='Agent ID to visualize')
    parser.add_argument('--output_dir', type=str, default='visualization_output',
                        help='Directory to save visualization outputs')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu or cuda)')
    parser.add_argument('--viz_type', type=str, default='all',
                        choices=['all', '2d', '3d_barrier', '3d_task', '3d_total', 'gradient', 'slice'],
                        help='Type of visualization to generate')
    parser.add_argument('--resolution', type=int, default=100,
                        help='Grid resolution for potential field computation')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = BarrierPotentialVisualizer(
        model_dir=args.model_dir,
        task=args.task,
        agent_id=args.agent_id,
        device=args.device
    )
    
    # Generate visualizations
    if args.viz_type == 'all':
        visualizer.visualize_all(output_dir=args.output_dir)
    elif args.viz_type == '2d':
        visualizer.visualize_potential_2d(resolution=args.resolution, save_path=args.output_dir)
    elif args.viz_type == '3d_barrier':
        visualizer.visualize_potential_3d(resolution=args.resolution, potential_type='barrier', 
                                          save_path=args.output_dir)
    elif args.viz_type == '3d_task':
        visualizer.visualize_potential_3d(resolution=args.resolution, potential_type='task',
                                          save_path=args.output_dir)
    elif args.viz_type == '3d_total':
        visualizer.visualize_potential_3d(resolution=args.resolution, potential_type='total',
                                          save_path=args.output_dir)
    elif args.viz_type == 'gradient':
        visualizer.visualize_gradient_field(resolution=20, save_path=args.output_dir)
    elif args.viz_type == 'slice':
        visualizer.visualize_slice_comparison(y_slice=0.0, resolution=200, save_path=args.output_dir)


if __name__ == '__main__':
    main()
