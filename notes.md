## Run
uv run safepo/single_agent/ppo_lag.py --task SafetyPointGoal1-v0 --experiment ppo_lag_exp

CUDA_VISIBLE_DEVICES=2 uv run safepo/multi_agent/macpo.py --task SafetyPointMultiGoal0-v0 --experiment macpo_exp