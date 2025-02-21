import gym
import highway_env
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from vehicles_model.vehicle_init_data import simulate_idm
import os
import torch

# # Clear GPU cache
# torch.cuda.empty_cache()

# # Set CUDA_LAUNCH_BLOCKING for debugging
# os.environ["CUDA_LAUNCH_BLOCKING"] = "0,1,2,3,4,5,6,7"

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# Create environment
env = gym.make("merge-v16")

# load model
model = DDPG.load("/data/wangzm/merge/Bench4Merge/Model_Hiera_RL.zip", env=env)
# model = DDPG.load("/data/wangzm/merge/Bench4Merge/Model_RL_DDPG.zip", env=env)

mean_reward, std_reward = evaluate_policy(
    model,
    model.get_env(),
    deterministic=True,
    render=True,
    n_eval_episodes=1)
print(mean_reward)