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
Barrier Potential Video Visualizer for MAPPO-Safe-PINN

This module provides real-time visualization of the learned barrier potential field
alongside environment rendering, creating side-by-side video comparisons.

Based on the VMAS potential visualizer implementation, adapted for SafetyMultiGoal
environments (Point/Car agents) with Barrier Port-Hamiltonian PINN actors.

Usage:
    visualizer = BarrierPotentialVideoVisualizer(
        actor=policy.actor,
        world_bounds=(-2.5, 2.5, -2.5, 2.5),
        device='cpu'
    )
    
    # During episode rollout
    frame = visualizer.render_combined_frame(
        env_frame=env.render(),
        obs=obs,
        agent_positions=agent_positions,
        obstacle_positions=obstacle_positions,
        goal_positions=goal_positions,
        step=step
    )
    frames.append(frame)
    
    # Save video
    visualizer.save_video(frames, output_path='potential_field_video.mp4')

Key Features:
    1. Real-time potential field computation from the learned actor
    2. Side-by-side environment and potential field visualization
    3. Agent/obstacle/goal overlays on potential field
    4. Video export with proper fps and codec
"""

import os
import numpy as np
import torch
from typing import Optional, Tuple, List, Dict, Any

# Make matplotlib optional
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server environments
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, LogNorm
    from matplotlib import cm
    from matplotlib.patches import Circle
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    Normalize = None
    LogNorm = None
    cm = None

# Make imageio optional for video export
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    imageio = None


class BarrierPotentialVideoVisualizer:
    """
    Visualizes barrier potential fields dynamically during evaluation rollouts.
    
    This visualizer computes the learned barrier potential field at each timestep
    and renders it alongside the environment frame for comparison.
    
    Args:
        actor: BarrierPHSPINNActor model for computing potentials
        world_bounds: Tuple of (x_min, x_max, y_min, y_max) for the visualization grid
        grid_resolution: Number of grid points in each dimension
        device: PyTorch device for computation
        hazard_radius: Radius of hazard obstacles (default 0.25 for MultiGoal1)
    """
    
    def __init__(
        self,
        actor=None,
        world_bounds: Tuple[float, float, float, float] = (-2.5, 2.5, -2.5, 2.5),
        grid_resolution: int = 50,
        device: str = "cpu",
        hazard_radius: float = 0.25,
    ):
        self.actor = actor
        self.world_bounds = world_bounds
        self.grid_resolution = grid_resolution
        self.device = device
        self.hazard_radius = hazard_radius
        
        # Pre-compute grid
        x = np.linspace(world_bounds[0], world_bounds[1], grid_resolution)
        y = np.linspace(world_bounds[2], world_bounds[3], grid_resolution)
        self.X, self.Y = np.meshgrid(x, y)
        self.grid_points = np.stack([self.X.flatten(), self.Y.flatten()], axis=1)
        
        # Color schemes for agents
        self.agent_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        
    def set_actor(self, actor):
        """Set or update the actor model."""
        self.actor = actor
        
    def compute_barrier_potential_field(
        self,
        obs: torch.Tensor,
        obstacle_positions: Optional[np.ndarray] = None,
        agent_positions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute the barrier potential field over the grid using the learned model.
        
        This method constructs synthetic observations for each grid point and
        computes the barrier potential using the actor's _compute_barrier_potential method.
        
        Args:
            obs: Current observation tensor [n_agents, obs_dim]
            obstacle_positions: Obstacle/hazard positions, shape (n_obstacles, 2)
            agent_positions: Agent positions, shape (n_agents, 2)
            
        Returns:
            potential_field: 2D numpy array of shape (grid_resolution, grid_resolution)
        """
        if self.actor is None:
            return np.zeros((self.grid_resolution, self.grid_resolution))
        
        # Get observation template from first agent
        if isinstance(obs, torch.Tensor):
            obs_template = obs[0].clone() if obs.dim() > 1 else obs.clone()
        else:
            obs_template = torch.tensor(obs[0] if len(obs.shape) > 1 else obs, 
                                        dtype=torch.float32, device=self.device)

        obs_dim = obs_template.shape[-1]
        potential_field = np.zeros((self.grid_resolution, self.grid_resolution))
        
        # Compute potential at each grid point
        with torch.no_grad():
            for i in range(self.grid_resolution):
                for j in range(self.grid_resolution):
                    x = self.X[i, j]
                    y = self.Y[i, j]
                    
                    # Create synthetic observation for this grid position
                    synth_obs = self._create_synthetic_observation(
                        x, y, obs_template.clone(), 
                        obstacle_positions, agent_positions
                    )
                    
                    # Compute barrier potential
                    H_barrier, _ = self.actor._compute_barrier_potential(synth_obs)
                    potential_field[i, j] = H_barrier.item()
        
        return potential_field
    
    def compute_barrier_potential_field_fast(
        self,
        obstacle_positions: Optional[np.ndarray] = None,
        agent_positions: Optional[np.ndarray] = None,
        current_agent_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute barrier potential field analytically (v5.0 - Highly Localized).
        
        This method directly computes the barrier potential from distances without
        going through the neural network, matching the actor's exponential formula.
        
        Key Features (v5.0):
        - Far from obstacle (proximity < 0.75): H = 0 (white)
        - Near obstacle edge (0.75 < proximity < 0.85): H = 0~3 (yellow)
        - Close to obstacle (0.85 < proximity < 0.95): H = 3~6 (orange/red)  
        - At obstacle center (proximity > 0.95): H = 6~10 (black)
        - Other agents are treated as dynamic obstacles with smaller radius
        
        Args:
            obstacle_positions: Obstacle/hazard positions, shape (n_obstacles, 2)
            agent_positions: Agent positions (treated as dynamic obstacles), shape (n_agents, 2)
            current_agent_idx: Index of the agent for which we're computing potential
                              (that agent is excluded from repellers). If None, all agents are repellers.
            
        Returns:
            potential_field: 2D numpy array of shape (grid_resolution, grid_resolution)
        """
        potential_field = np.zeros((self.grid_resolution, self.grid_resolution))
        
        # v5.0 parameters (must match phs_mappo_actor.py)
        activation_threshold = 0.75  # Only activate when very close
        alpha = 4.0  # Exponential growth rate (smoother)
        scale = 10.0  # Maximum barrier value at center
        
        # Define lidar range (matches safety_gymnasium pseudo lidar)
        max_lidar_dist = 3.0  # Maximum lidar detection distance
        
        # Agent collision radius (smaller than obstacle hazard radius)
        agent_radius = 0.17  # r_collision from actor config
        
        # Collect all repellers with their radii
        repellers = []  # List of (position, radius)
        
        # Static obstacles (hazards)
        if obstacle_positions is not None and len(obstacle_positions) > 0:
            for pos in obstacle_positions:
                repellers.append((np.array(pos), self.hazard_radius))
        
        # Other agents as dynamic obstacles
        if agent_positions is not None and len(agent_positions) > 0:
            for idx, pos in enumerate(agent_positions):
                # Skip current agent (don't treat self as obstacle)
                if current_agent_idx is not None and idx == current_agent_idx:
                    continue
                repellers.append((np.array(pos), agent_radius))
        
        if len(repellers) == 0:
            return potential_field
        
        # Precompute exp(alpha)
        exp_alpha = np.exp(alpha)
        
        # Effective range after threshold
        effective_range = 1.0 - activation_threshold  # 0.25
        
        # Vectorized computation over grid
        for i in range(self.grid_resolution):
            for j in range(self.grid_resolution):
                pos = np.array([self.X[i, j], self.Y[i, j]])
                
                # Find minimum distance considering each repeller's radius
                min_dist = float('inf')
                for rep_pos, rep_radius in repellers:
                    dist = np.linalg.norm(rep_pos - pos) - rep_radius
                    min_dist = min(min_dist, max(dist, 0.0))
                
                # Convert distance to proximity (like lidar)
                # proximity = 1 when at obstacle surface, 0 when far away
                proximity = np.clip(1.0 - min_dist / max_lidar_dist, 0.0, 1.0)
                
                # v5.0: Highly localized exponential barrier (matches actor)
                if proximity < activation_threshold:
                    # Far from obstacle: zero potential (white in visualization)
                    H_barrier = 0.0
                else:
                    # Shift and normalize proximity to [0, 1] range
                    shifted_proximity = np.clip(
                        (proximity - activation_threshold) / effective_range,
                        0.0, 1.0
                    )
                    
                    # Exponential barrier: H = scale * (exp(α * shifted) - 1) / (exp(α) - 1)
                    exp_term = np.exp(np.minimum(alpha * shifted_proximity, 8.0))
                    H_barrier = scale * (exp_term - 1.0) / (exp_alpha - 1.0)
                    
                    # Clamp to max scale
                    H_barrier = np.minimum(H_barrier, scale)
                
                potential_field[i, j] = H_barrier
        
        return potential_field
    
    def compute_task_potential_field_fast(
        self,
        goal_positions: Optional[np.ndarray] = None,
        task_potential_scale: float = 2.0,
    ) -> np.ndarray:
        """
        Compute task potential field analytically.
        
        Task potential is attractive toward goals:
        H_task = scale * distance_to_nearest_goal
        
        This creates a potential well (low/negative values) at goal positions,
        guiding agents toward them.
        
        Args:
            goal_positions: Goal positions, shape (n_goals, 2)
            task_potential_scale: Scale factor for task potential
            
        Returns:
            potential_field: 2D numpy array of shape (grid_resolution, grid_resolution)
        """
        potential_field = np.zeros((self.grid_resolution, self.grid_resolution))
        
        if goal_positions is None or len(goal_positions) == 0:
            return potential_field
        
        goal_arr = np.array(goal_positions)
        
        # Compute task potential at each grid point
        for i in range(self.grid_resolution):
            for j in range(self.grid_resolution):
                pos = np.array([self.X[i, j], self.Y[i, j]])
                
                # Distance to all goals
                dists = np.linalg.norm(goal_arr - pos, axis=1)
                
                # Task potential: negative near goals (attractive)
                # Use negative exponential for attractive potential
                # H_task = min_dist (linear gradient toward goal)
                # Or use exponential for smoother visualization
                min_dist = np.min(dists)
                
                # Option 1: Linear potential (gradient field)
                # H_task = task_potential_scale * min_dist
                
                # Option 2: Negative Gaussian (creates potential well)
                # H_task = -task_potential_scale * exp(-min_dist^2 / (2 * sigma^2))
                sigma = 1.0  # Width of the potential well
                H_task = -task_potential_scale * np.exp(-min_dist**2 / (2 * sigma**2))
                
                potential_field[i, j] = H_task
        
        return potential_field
    
    def compute_total_potential_field_fast(
        self,
        obstacle_positions: Optional[np.ndarray] = None,
        goal_positions: Optional[np.ndarray] = None,
        agent_positions: Optional[np.ndarray] = None,
        task_potential_scale: float = 2.0,
    ) -> np.ndarray:
        """
        Compute total potential field: H_total = H_barrier + H_task
        
        This combines:
        - Barrier potential (repulsive from obstacles) 
        - Task potential (attractive to goals)
        
        Args:
            obstacle_positions: Obstacle/hazard positions
            goal_positions: Goal positions
            agent_positions: Agent positions (optional repellers)
            task_potential_scale: Scale for task potential
            
        Returns:
            potential_field: 2D numpy array of shape (grid_resolution, grid_resolution)
        """
        H_barrier = self.compute_barrier_potential_field_fast(
            obstacle_positions=obstacle_positions,
            agent_positions=agent_positions,
        )
        
        H_task = self.compute_task_potential_field_fast(
            goal_positions=goal_positions,
            task_potential_scale=task_potential_scale,
        )
        
        # Total potential: barrier (positive/repulsive) + task (negative/attractive)
        H_total = H_barrier + H_task
        
        return H_total
    
    def _create_synthetic_observation(
        self,
        x: float, 
        y: float, 
        obs_template: torch.Tensor,
        obstacle_positions: Optional[np.ndarray] = None,
        agent_positions: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Create synthetic observation for a given grid position.
        
        Updates the lidar observations based on distances from (x, y) to obstacles.
        
        Observation structure (MultiGoal):
        - obs[0:3]: accelerometer (ax, ay, az)
        - obs[3:6]: velocimeter (vx, vy, vz)
        - obs[6:9]: gyro
        - obs[9:12]: magnetometer  
        - obs[12:28]: goal_red lidar (16 bins)
        - obs[28:44]: goal_blue lidar (16 bins)
        - obs[44:60]: hazard lidar (16 bins) - CRITICAL for barrier
        """
        obs = obs_template.unsqueeze(0)  # [1, obs_dim]
        
        if obstacle_positions is None or len(obstacle_positions) == 0:
            return obs
        
        # Update hazard lidar (obs[44:60])
        num_bins = 16
        angles = np.linspace(0, 2*np.pi, num_bins, endpoint=False)
        agent_pos = np.array([x, y])
        
        hazard_lidar = self._compute_lidar_readings(
            agent_pos, obstacle_positions, angles, 
            max_dist=3.0, radius=self.hazard_radius
        )
        
        # Update observation
        obs_np = obs.cpu().numpy().copy()
        if obs_np.shape[1] > 60:
            obs_np[0, 44:60] = hazard_lidar
        
        return torch.tensor(obs_np, dtype=torch.float32, device=self.device)
    
    def _compute_lidar_readings(
        self,
        agent_pos: np.ndarray,
        obstacle_positions: np.ndarray,
        angles: np.ndarray,
        max_dist: float = 3.0,
        radius: float = 0.25,
    ) -> np.ndarray:
        """
        Compute lidar readings based on distances to obstacles.
        
        Uses exponential decay: closer obstacles = higher readings.
        """
        num_bins = len(angles)
        lidar = np.zeros(num_bins, dtype=np.float32)
        
        for i, angle in enumerate(angles):
            ray_dir = np.array([np.cos(angle), np.sin(angle)])
            min_dist = max_dist
            
            for obs_pos in obstacle_positions:
                obs_pos = np.array(obs_pos)
                to_obs = obs_pos - agent_pos
                
                # Project onto ray direction
                proj_dist = np.dot(to_obs, ray_dir)
                
                if proj_dist > 0:
                    perp_dist = np.abs(np.cross(ray_dir, to_obs))
                    
                    if perp_dist < radius:
                        hit_dist = proj_dist - np.sqrt(max(0, radius**2 - perp_dist**2))
                        min_dist = min(min_dist, max(0, hit_dist))
            
            # Exponential decay
            if min_dist < max_dist:
                lidar[i] = np.exp(-min_dist)
        
        return lidar
    
    def render_combined_frame(
        self,
        env_frame: np.ndarray,
        potential_field: np.ndarray,
        agent_positions: Optional[np.ndarray] = None,
        obstacle_positions: Optional[np.ndarray] = None,
        goal_positions: Optional[np.ndarray] = None,
        step: int = 0,
        figsize: Tuple[int, int] = (16, 8),
        dpi: int = 100,
    ) -> np.ndarray:
        """
        Render combined visualization: environment frame + barrier potential field.
        
        Args:
            env_frame: RGB image from environment render
            potential_field: 2D array of potential values
            agent_positions: Positions of agents, shape (n_agents, 2)
            obstacle_positions: Positions of obstacles, shape (n_obstacles, 2)
            goal_positions: Goal positions, shape (n_goals, 2)
            step: Current step number
            figsize: Figure size in inches
            dpi: Resolution
            
        Returns:
            combined_image: Combined RGB image as numpy array
        """
        if not HAS_MATPLOTLIB:
            # Return concatenated frames if matplotlib unavailable
            h, w = env_frame.shape[:2]
            return np.zeros((h, w * 2, 3), dtype=np.uint8)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        
        # === Left: Environment Frame ===
        axes[0].imshow(env_frame)
        axes[0].set_title(f'Environment (Step {step})', fontsize=14)
        axes[0].axis('off')
        
        # === Right: Barrier Potential Field ===
        # Normalize potential field
        if (potential_field > 0).any():
            vmax = np.percentile(potential_field[potential_field > 0], 95)
        else:
            vmax = 1.0
        vmin = 0
        norm = Normalize(vmin=vmin, vmax=max(vmax, 0.1))
        
        extent = [self.world_bounds[0], self.world_bounds[1], 
                  self.world_bounds[2], self.world_bounds[3]]
        
        im = axes[1].imshow(
            potential_field, 
            extent=extent, 
            origin='lower',
            cmap='hot_r',
            norm=norm,
            alpha=0.8
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1], shrink=0.8)
        cbar.set_label('Barrier Potential', fontsize=10)
        
        # Plot obstacles
        if obstacle_positions is not None and len(obstacle_positions) > 0:
            obs_arr = np.array(obstacle_positions)
            axes[1].scatter(
                obs_arr[:, 0], obs_arr[:, 1],
                c='gray', s=200, marker='o', 
                edgecolors='black', linewidths=2,
                label='Obstacles', zorder=5
            )
            # Draw obstacle circles
            for pos in obstacle_positions:
                circle = Circle(pos, self.hazard_radius, 
                              fill=False, color='gray', linewidth=1.5, linestyle='--')
                axes[1].add_patch(circle)
        
        # Plot agents (only if within bounds)
        if agent_positions is not None and len(agent_positions) > 0:
            agent_arr = np.array(agent_positions)
            n_agents = len(agent_arr)
            colors = [self.agent_colors[i % len(self.agent_colors)] for i in range(n_agents)]
            
            x_min, x_max, y_min, y_max = self.world_bounds
            for i, (pos, color) in enumerate(zip(agent_arr, colors)):
                # Only plot agents within world bounds
                if x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max:
                    axes[1].scatter(
                        pos[0], pos[1],
                        c=color, s=150, marker='o',
                        edgecolors='black', linewidths=2,
                        label=f'Agent {i}' if i < 2 else '', zorder=6
                    )
        
        # Plot goals
        if goal_positions is not None and len(goal_positions) > 0:
            goal_arr = np.array(goal_positions)
            for i, pos in enumerate(goal_arr):
                color = self.agent_colors[i % len(self.agent_colors)]
                axes[1].scatter(
                    pos[0], pos[1],
                    c=color, s=100, marker='*',
                    edgecolors='black', linewidths=1,
                    label=f'Goal {i}' if i < 2 else '', zorder=5
                )
        
        axes[1].set_xlabel('X Position', fontsize=12)
        axes[1].set_ylabel('Y Position', fontsize=12)
        axes[1].set_title('Learned Barrier Potential Field', fontsize=14)
        
        # Fix axis bounds
        axes[1].set_xlim(self.world_bounds[0], self.world_bounds[1])
        axes[1].set_ylim(self.world_bounds[2], self.world_bounds[3])
        
        # Add legend
        handles, labels = axes[1].get_legend_handles_labels()
        if handles:
            axes[1].legend(loc='upper right', fontsize=9)
        
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to numpy array
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        
        plt.close(fig)
        
        return image
    
    def render_all_potentials_frame(
        self,
        env_frame: np.ndarray,
        obstacle_positions: Optional[np.ndarray] = None,
        goal_positions: Optional[np.ndarray] = None,
        agent_positions: Optional[np.ndarray] = None,
        step: int = 0,
        task_potential_scale: float = 2.0,
        figsize: Tuple[int, int] = (20, 5),
        dpi: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Render all three potential fields: H_barrier, H_task, H_total
        
        Returns combined visualization frame and individual potential field images.
        
        Args:
            env_frame: RGB image from environment render
            obstacle_positions: Positions of obstacles
            goal_positions: Goal positions
            agent_positions: Agent positions
            step: Current step number
            task_potential_scale: Scale for task potential
            figsize: Figure size in inches
            dpi: Resolution
            
        Returns:
            Tuple of (combined_image, barrier_image, task_image, total_image)
        """
        if not HAS_MATPLOTLIB:
            h, w = env_frame.shape[:2]
            empty = np.zeros((h, w, 3), dtype=np.uint8)
            return empty, empty, empty, empty
        
        # Compute all potential fields
        H_barrier = self.compute_barrier_potential_field_fast(
            obstacle_positions=obstacle_positions,
            agent_positions=agent_positions,
        )
        H_task = self.compute_task_potential_field_fast(
            goal_positions=goal_positions,
            task_potential_scale=task_potential_scale,
        )
        H_total = H_barrier + H_task
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(1, 4, figsize=figsize, dpi=dpi)
        
        extent = [self.world_bounds[0], self.world_bounds[1], 
                  self.world_bounds[2], self.world_bounds[3]]
        
        # === 1. Environment Frame ===
        axes[0].imshow(env_frame)
        axes[0].set_title(f'Environment (Step {step})', fontsize=12)
        axes[0].axis('off')
        
        # === 2. Barrier Potential ===
        vmax_barrier = np.percentile(H_barrier, 95) if H_barrier.max() > 0 else 1.0
        im1 = axes[1].imshow(
            H_barrier, extent=extent, origin='lower',
            cmap='hot_r', vmin=0, vmax=max(vmax_barrier, 0.1), alpha=0.8,
            interpolation='bicubic'
        )
        self._add_overlays(axes[1], obstacle_positions, goal_positions, agent_positions)
        axes[1].set_title('Barrier Potential (V_barrier)', fontsize=12)
        axes[1].set_xlabel('X Position')
        axes[1].set_ylabel('Y Position')
        axes[1].set_xlim(self.world_bounds[0], self.world_bounds[1])
        axes[1].set_ylim(self.world_bounds[2], self.world_bounds[3])
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        cbar1 = plt.colorbar(im1, ax=axes[1], shrink=0.8)
        cbar1.set_label('V_barrier', fontsize=9)
        
        # === 3. Task Potential ===
        vmin_task = H_task.min()
        vmax_task = max(H_task.max(), 0.1)
        im2 = axes[2].imshow(
            H_task, extent=extent, origin='lower',
            cmap='coolwarm', vmin=vmin_task, vmax=vmax_task, alpha=0.8,
            interpolation='bicubic'
        )
        self._add_overlays(axes[2], obstacle_positions, goal_positions, agent_positions)
        axes[2].set_title('Task Potential (V_task)', fontsize=12)
        axes[2].set_xlabel('X Position')
        axes[2].set_ylabel('Y Position')
        axes[2].set_xlim(self.world_bounds[0], self.world_bounds[1])
        axes[2].set_ylim(self.world_bounds[2], self.world_bounds[3])
        axes[2].set_aspect('equal')
        axes[2].grid(True, alpha=0.3)
        cbar2 = plt.colorbar(im2, ax=axes[2], shrink=0.8)
        cbar2.set_label('V_task', fontsize=9)
        
        # === 4. Total Potential ===
        vmin_total = H_total.min()
        vmax_total = np.percentile(H_total, 95) if H_total.max() > 0 else 1.0
        im3 = axes[3].imshow(
            H_total, extent=extent, origin='lower',
            cmap='RdYlGn_r', vmin=vmin_total, vmax=max(vmax_total, abs(vmin_total)), alpha=0.8,
            interpolation='bicubic'
        )
        self._add_overlays(axes[3], obstacle_positions, goal_positions, agent_positions)
        axes[3].set_title('Total Potential (V_total)', fontsize=12)
        axes[3].set_xlabel('X Position')
        axes[3].set_ylabel('Y Position')
        axes[3].set_xlim(self.world_bounds[0], self.world_bounds[1])
        axes[3].set_ylim(self.world_bounds[2], self.world_bounds[3])
        axes[3].set_aspect('equal')
        axes[3].grid(True, alpha=0.3)
        cbar3 = plt.colorbar(im3, ax=axes[3], shrink=0.8)
        cbar3.set_label('V_total', fontsize=9)
        
        plt.tight_layout()
        
        # Convert to numpy array
        fig.canvas.draw()
        combined_image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        
        # Generate individual potential images for wandb logging
        barrier_image = self._render_single_potential(H_barrier, 'hot_r', 'Barrier Potential', 
                                                       obstacle_positions, goal_positions, agent_positions, step)
        task_image = self._render_single_potential(H_task, 'coolwarm', 'Task Potential',
                                                    obstacle_positions, goal_positions, agent_positions, step)
        total_image = self._render_single_potential(H_total, 'RdYlGn_r', 'Total Potential',
                                                     obstacle_positions, goal_positions, agent_positions, step)
        
        return combined_image, barrier_image, task_image, total_image
    
    def _add_overlays(self, ax, obstacle_positions, goal_positions, agent_positions):
        """Add obstacles, goals, and agents to axis."""
        # Plot obstacles
        if obstacle_positions is not None and len(obstacle_positions) > 0:
            obs_arr = np.array(obstacle_positions)
            ax.scatter(obs_arr[:, 0], obs_arr[:, 1],
                      c='gray', s=100, marker='o', edgecolors='black', linewidths=1.5, zorder=5)
            for pos in obstacle_positions:
                circle = Circle(pos, self.hazard_radius, 
                              fill=False, color='gray', linewidth=1, linestyle='--')
                ax.add_patch(circle)
        
        # Plot agents
        if agent_positions is not None and len(agent_positions) > 0:
            agent_arr = np.array(agent_positions)
            for i, pos in enumerate(agent_arr):
                color = self.agent_colors[i % len(self.agent_colors)]
                ax.scatter(pos[0], pos[1], c=color, s=100, marker='o',
                          edgecolors='black', linewidths=1.5, zorder=6)
        
        # Plot goals
        if goal_positions is not None and len(goal_positions) > 0:
            goal_arr = np.array(goal_positions)
            for i, pos in enumerate(goal_arr):
                color = self.agent_colors[i % len(self.agent_colors)]
                ax.scatter(pos[0], pos[1], c=color, s=80, marker='*',
                          edgecolors='black', linewidths=1, zorder=5)
    
    def _render_single_potential(self, potential_field, cmap, title, 
                                  obstacle_positions, goal_positions, agent_positions, step,
                                  figsize=(6, 6), dpi=80) -> np.ndarray:
        """Render a single potential field image for wandb logging."""
        if not HAS_MATPLOTLIB:
            return np.zeros((480, 480, 3), dtype=np.uint8)
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        extent = [self.world_bounds[0], self.world_bounds[1], 
                  self.world_bounds[2], self.world_bounds[3]]
        
        vmin = potential_field.min()
        vmax = np.percentile(potential_field, 95) if potential_field.max() > abs(vmin) else abs(vmin)
        
        im = ax.imshow(
            potential_field, extent=extent, origin='lower',
            cmap=cmap, vmin=vmin, vmax=max(vmax, 0.1), alpha=0.8,
            interpolation='bicubic'
        )
        
        self._add_overlays(ax, obstacle_positions, goal_positions, agent_positions)
        
        ax.set_title(f'{title} (Step {step})', fontsize=12)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_xlim(self.world_bounds[0], self.world_bounds[1])
        ax.set_ylim(self.world_bounds[2], self.world_bounds[3])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(title.split()[0], fontsize=9)
        
        plt.tight_layout()
        
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        
        return image
    
    def save_video(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: int = 30,
    ) -> bool:
        """
        Save frames as video file.
        
        Args:
            frames: List of RGB images
            output_path: Path to output video file
            fps: Frames per second
            
        Returns:
            success: Whether the video was saved successfully
        """
        if not HAS_IMAGEIO:
            print("[BarrierViz] imageio not available, cannot save video")
            return False
        
        if len(frames) == 0:
            print("[BarrierViz] No frames to save")
            return False
        
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write video with proper error handling
            try:
                # Use macro_block_size=1 to avoid resizing warnings
                writer = imageio.get_writer(output_path, fps=fps, codec='libx264', macro_block_size=1)
                for frame in frames:
                    writer.append_data(frame)
                writer.close()
            except Exception as write_error:
                # Attempt cleanup even if write fails
                try:
                    writer.close()
                except:
                    pass
                raise write_error
            
            return True
            
        except Exception as e:
            print(f"[BarrierViz] Failed to save video: {e}")
            return False
        finally:
            # Always clean up matplotlib resources
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
            except:
                pass
    
    def upload_video_to_wandb(
        self,
        frames: List[np.ndarray],
        step: int,
        key: str = "eval/potential_field_video",
        caption: str = "",
        fps: int = 30,
    ) -> bool:
        """
        Upload video frames to wandb.
        
        Args:
            frames: List of RGB images
            step: Current training step
            key: wandb logging key
            caption: Video caption
            fps: Frames per second
            
        Returns:
            success: Whether the upload was successful
        """
        try:
            import wandb
            
            if wandb.run is None:
                print("[BarrierViz] No active wandb run")
                return False
            
            if len(frames) == 0:
                print("[BarrierViz] No frames to upload")
                return False
            
            # Convert frames to (T, H, W, C) format
            video_array = np.stack(frames, axis=0)
            
            # Create wandb Video object
            video = wandb.Video(video_array, fps=fps, format="mp4", caption=caption)
            
            # Log to wandb
            wandb.log({key: video}, step=step)
            
            print(f"[BarrierViz] Video uploaded to wandb at step {step}")
            return True
            
        except Exception as e:
            print(f"[BarrierViz] Failed to upload video to wandb: {e}")
            return False


class EvalPotentialFieldRecorder:
    """
    Helper class for recording potential field visualizations during evaluation.
    
    This class manages the recording process and provides a simple interface
    for integrating with the MAPPO-Safe-PINN eval loop.
    """
    
    def __init__(
        self,
        actor,
        world_bounds: Tuple[float, float, float, float] = (-2.5, 2.5, -2.5, 2.5),
        grid_resolution: int = 50,
        device: str = "cpu",
        hazard_radius: float = 0.25,
        output_dir: str = "./vizs",
    ):
        self.visualizer = BarrierPotentialVideoVisualizer(
            actor=actor,
            world_bounds=world_bounds,
            grid_resolution=grid_resolution,
            device=device,
            hazard_radius=hazard_radius,
        )
        self.output_dir = output_dir
        self.frames = []
        self.step_count = 0
        
        os.makedirs(output_dir, exist_ok=True)
    
    def reset(self):
        """Reset for new episode."""
        self.frames = []
        self.step_count = 0
    
    def record_step(
        self,
        env_frame: np.ndarray,
        agent_positions: np.ndarray,
        obstacle_positions: np.ndarray,
        goal_positions: np.ndarray,
        obs: Optional[torch.Tensor] = None,
        use_fast_compute: bool = True,
    ):
        """
        Record a single step.
        
        Args:
            env_frame: Environment render frame
            agent_positions: Agent positions
            obstacle_positions: Obstacle positions
            goal_positions: Goal positions
            obs: Current observation (optional, for slow but accurate computation)
            use_fast_compute: If True, use analytical barrier computation (faster)
        """
        # Compute potential field
        if use_fast_compute:
            potential_field = self.visualizer.compute_barrier_potential_field_fast(
                obstacle_positions=obstacle_positions,
                agent_positions=agent_positions,
            )
        else:
            potential_field = self.visualizer.compute_barrier_potential_field(
                obs=obs,
                obstacle_positions=obstacle_positions,
                agent_positions=agent_positions,
            )
        
        # Render combined frame
        frame = self.visualizer.render_combined_frame(
            env_frame=env_frame,
            potential_field=potential_field,
            agent_positions=agent_positions,
            obstacle_positions=obstacle_positions,
            goal_positions=goal_positions,
            step=self.step_count,
        )
        
        self.frames.append(frame)
        self.step_count += 1
    
    def save_video(self, filename: str = "potential_field_video.mp4", fps: int = 30) -> str:
        """Save recorded frames as video."""
        output_path = os.path.join(self.output_dir, filename)
        self.visualizer.save_video(self.frames, output_path, fps=fps)
        return output_path
    
    def upload_to_wandb(
        self,
        step: int,
        key: str = "eval/potential_field_video",
        caption: str = "",
        fps: int = 30,
    ) -> bool:
        """Upload to wandb."""
        return self.visualizer.upload_video_to_wandb(
            frames=self.frames,
            step=step,
            key=key,
            caption=caption,
            fps=fps,
        )
    
    def get_frames(self) -> List[np.ndarray]:
        """Get recorded frames."""
        return self.frames
