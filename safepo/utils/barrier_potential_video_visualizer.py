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
    ) -> np.ndarray:
        """
        Compute barrier potential field analytically (faster, uses distance-based formula).
        
        This method directly computes the barrier potential from distances without
        going through the neural network, providing a faster visualization.
        
        H_barrier = k / ((d - r_safe)^2 + ε)
        
        Args:
            obstacle_positions: Obstacle/hazard positions, shape (n_obstacles, 2)
            agent_positions: Agent positions, shape (n_agents, 2)
            
        Returns:
            potential_field: 2D numpy array of shape (grid_resolution, grid_resolution)
        """
        potential_field = np.zeros((self.grid_resolution, self.grid_resolution))
        
        # Get barrier parameters from actor if available
        if self.actor is not None:
            r_safe = getattr(self.actor, 'r_safe', 0.3)
            barrier_epsilon = getattr(self.actor, 'barrier_epsilon', 0.01)
            barrier_k = getattr(self.actor, 'barrier_k_scale', 2.0)
        else:
            r_safe = 0.3
            barrier_epsilon = 0.01
            barrier_k = 2.0
        
        # Combine obstacles and agents as repellers
        all_repellers = []
        if obstacle_positions is not None and len(obstacle_positions) > 0:
            all_repellers.extend(obstacle_positions)
        if agent_positions is not None and len(agent_positions) > 0:
            all_repellers.extend(agent_positions)
        
        if len(all_repellers) == 0:
            return potential_field
        
        all_repellers = np.array(all_repellers)
        
        # Vectorized computation over grid
        for i in range(self.grid_resolution):
            for j in range(self.grid_resolution):
                pos = np.array([self.X[i, j], self.Y[i, j]])
                
                # Distance to all repellers
                dists = np.linalg.norm(all_repellers - pos, axis=1)
                
                # Barrier potential: k / ((d - r_safe)^2 + ε)
                # Use hazard_radius for obstacles
                margin = np.maximum(dists - self.hazard_radius - r_safe, 0.01)
                H_barrier = barrier_k / (margin ** 2 + barrier_epsilon)
                
                # Sum over all repellers
                potential_field[i, j] = np.sum(H_barrier)
        
        return potential_field
    
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
                writer = imageio.get_writer(output_path, fps=fps, codec='libx264')
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
