# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
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

"""Video recording and WandB upload utilities for headless servers."""

import os
import numpy as np
from typing import List, Optional
import tempfile

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, video logging will be disabled")

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("Warning: imageio not available, video recording will be disabled")


class VideoRecorder:
    """
    Records environment frames and uploads videos to WandB without saving locally.
    Designed for headless servers with no display.
    """

    def __init__(self, fps: int = 30, enabled: bool = True):
        """
        Initialize the video recorder.

        Args:
            fps: Frames per second for the video
            enabled: Whether video recording is enabled
        """
        self.fps = fps
        self.enabled = enabled and WANDB_AVAILABLE and IMAGEIO_AVAILABLE
        self.frames: List[np.ndarray] = []
        
        if not WANDB_AVAILABLE:
            print("Warning: WandB not available, disabling video recording")
        if not IMAGEIO_AVAILABLE:
            print("Warning: imageio not available, disabling video recording")

    def capture_frame(self, frame: np.ndarray) -> None:
        """
        Capture a single frame.

        Args:
            frame: RGB frame as numpy array (H, W, 3)
        """
        if self.enabled and frame is not None:
            self.frames.append(frame.copy())

    def clear(self) -> None:
        """Clear all captured frames."""
        self.frames = []

    def upload_to_wandb(
        self,
        caption: str = "Episode Video",
        step: Optional[int] = None,
        key: str = "video"
    ) -> bool:
        """
        Upload recorded video to WandB without saving locally.

        Args:
            caption: Caption for the video
            step: Training step number
            key: Key for logging to WandB

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or len(self.frames) == 0:
            return False

        try:
            # Create temporary file for video
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                tmp_path = tmp_file.name

            # Write video to temporary file
            writer = imageio.get_writer(tmp_path, fps=self.fps)
            for frame in self.frames:
                writer.append_data(frame)
            writer.close()

            # Upload to wandb
            if wandb.run is not None:
                wandb.log({
                    key: wandb.Video(tmp_path, caption=caption, fps=self.fps, format="mp4")
                }, step=step)
                
            # Clean up temporary file
            os.remove(tmp_path)
            
            # Clear frames after upload
            self.clear()
            return True

        except Exception as e:
            print(f"Error uploading video to wandb: {e}")
            # Clean up temporary file if it exists
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False

    def get_frame_count(self) -> int:
        """Get the number of captured frames."""
        return len(self.frames)


class MultiAgentVideoRecorder:
    """
    Manages video recording for multi-agent environments.
    Records multiple episodes and uploads to WandB periodically.
    """

    def __init__(
        self,
        fps: int = 30,
        enabled: bool = True,
        record_freq: int = 10,  # Record every N episodes
        max_episode_length: int = 1000
    ):
        """
        Initialize multi-agent video recorder.

        Args:
            fps: Frames per second
            enabled: Whether recording is enabled
            record_freq: Record video every N episodes
            max_episode_length: Maximum episode length for recording
        """
        self.recorder = VideoRecorder(fps=fps, enabled=enabled)
        self.enabled = enabled
        self.record_freq = record_freq
        self.max_episode_length = max_episode_length
        self.episode_count = 0
        self.is_recording = False

    def should_record(self) -> bool:
        """Check if current episode should be recorded."""
        return self.enabled and (self.episode_count % self.record_freq == 0)

    def start_episode(self) -> None:
        """Start recording a new episode if appropriate."""
        self.episode_count += 1
        self.is_recording = self.should_record()
        if self.is_recording:
            self.recorder.clear()

    def capture_frame(self, frame: np.ndarray) -> None:
        """
        Capture frame if currently recording.

        Args:
            frame: RGB frame from environment
        """
        if self.is_recording:
            # Limit episode length to avoid memory issues
            if self.recorder.get_frame_count() < self.max_episode_length:
                self.recorder.capture_frame(frame)

    def end_episode(
        self,
        episode_reward: float,
        episode_cost: float,
        step: Optional[int] = None
    ) -> bool:
        """
        End current episode and upload video if recording.

        Args:
            episode_reward: Total episode reward
            episode_cost: Total episode cost
            step: Training step number

        Returns:
            True if video was uploaded, False otherwise
        """
        if self.is_recording and self.recorder.get_frame_count() > 0:
            caption = f"Episode {self.episode_count} - Reward: {episode_reward:.2f}, Cost: {episode_cost:.2f}"
            success = self.recorder.upload_to_wandb(
                caption=caption,
                step=step,
                key="eval/video"
            )
            self.is_recording = False
            return success
        return False


def setup_headless_rendering():
    """
    Setup environment for headless rendering on servers without display.
    Sets appropriate environment variables for MuJoCo rendering.
    """
    import os
    
    # Use EGL for headless rendering (works on most GPU servers)
    os.environ['MUJOCO_GL'] = 'egl'
    
    # Alternative: use OSMesa for CPU-only rendering (slower)
    # os.environ['MUJOCO_GL'] = 'osmesa'
    
    # Disable display
    if 'DISPLAY' in os.environ:
        del os.environ['DISPLAY']
    
    print("Headless rendering configured with EGL backend")
