## Run
unset VIRTUAL_ENV

uv run safepo/single_agent/ppo_lag.py --task SafetyPointGoal1-v0 --device "cuda" --experiment ppo_lag_exp

CUDA_VISIBLE_DEVICES=2 uv run safepo/multi_agent/macpo.py --task SafetyPointMultiGoal0-v0 --device "cuda" --experiment macpo_exp

## Multi-Agent MultiGoal Environments (with video rendering support)
# Point
CUDA_VISIBLE_DEVICES=2 uv run safepo/multi_agent/macpo.py --task SafetyPointMultiGoal0-v0 --device "cuda" --experiment macpo_exp

# Car
CUDA_VISIBLE_DEVICES=2 uv run safepo/multi_agent/macpo.py --task SafetyCarMultiGoal0-v0 --device "cuda" --experiment macpo_exp

# Racecar
CUDA_VISIBLE_DEVICES=2 uv run safepo/multi_agent/macpo.py --task SafetyRacecarMultiGoal0-v0 --device "cuda" --experiment macpo_exp

# Doggo
CUDA_VISIBLE_DEVICES=2 uv run safepo/multi_agent/macpo.py --task SafetyDoggoMultiGoal0-v0 --device "cuda" --experiment macpo_exp

# Ant
CUDA_VISIBLE_DEVICES=2 uv run safepo/multi_agent/macpo.py --task SafetyAntMultiGoal0-v0 --device "cuda" --experiment macpo_exp

## Multi-Agent Velocity Environments (with video rendering support)
# Ant - 2x4 agents
CUDA_VISIBLE_DEVICES=2 uv run safepo/multi_agent/macpo.py --task Safety2x4AntVelocity-v0 --device "cuda" --experiment velocity_exp

# HalfCheetah - 6x1 agents
CUDA_VISIBLE_DEVICES=2 uv run safepo/multi_agent/macpo.py --task Safety6x1HalfCheetahVelocity-v0 --device "cuda" --experiment velocity_exp

# Humanoid - 9|8 agents
CUDA_VISIBLE_DEVICES=2 uv run safepo/multi_agent/macpo.py --task Safety9|8HumanoidVelocity-v0 --device "cuda" --experiment velocity_exp

## Video Rendering
# Both MultiGoal and Velocity environments support automatic video recording during evaluation
# Videos are saved to the experiment directory and uploaded to wandb
# Configure in config files:
#   - record_video: true/false
#   - video_record_freq: record every N evaluations
#   - render_mode: 'rgb_array' for offscreen rendering