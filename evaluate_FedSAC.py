import utils.setup_path
import gym
import airgym
import time
from torch import nn
import torch as th

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, StopTrainingOnMaxEpisodes
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecFrameStack

from utils.EpisodeCheckpointCallback import EpisodeCheckpointCallback

import argparse
import json
from subprocess import Popen
import numpy as np
import os
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--track", help="which track will be used, 0~2", type=int, default=1)
parser.add_argument("-i", "--intersection", help="which intersection car is in", type=int, default=1)
parser.add_argument("-l", "--log_name", help="modified log name", type=str, nargs='?')
args = parser.parse_args()

with open("settings.json") as f:
    settings = json.load(f)
settings["ViewMode"] = "SpringArmChase"
#settings["ViewMode"] = "NoDisplay"
Car = settings["Vehicles"]["Car1"]
if args.track == 1:
    if args.intersection == 1:
        Car["X"], Car["Y"], Car["Z"], Car["Yaw"] = (0, 0, 0, 180)
    elif args.intersection == 2:
        Car["X"], Car["Y"], Car["Z"], Car["Yaw"] = (-127, 0, 0, 270)
    elif args.intersection == 3:
        Car["X"], Car["Y"], Car["Z"], Car["Yaw"] = (-127, -128, 0, 0)
    elif args.intersection == 4:
        Car["X"], Car["Y"], Car["Z"], Car["Yaw"] = (0, -128, 0, 90)
    else:
        Car["X"], Car["Y"], Car["Z"], Car["Yaw"] = (0, 0, 0, 180)
settings["Vehicles"]["Car1"] = Car
with open("settings.json", "w") as f:
    json.dump(settings, f, indent=4)

print(Popen("./Environment.sh"))
time.sleep(7) #wait for airsim opening"

# Create a DummyVecEnv for main airsim gym env
env = gym.make(
                "airgym:airsim-car-cont-action-sample-v0",
                ip_address="127.0.0.1",
                image_shape=(84, 84, 1),
            )
env.env.setkwargs(track = args.track)
env.env.setInitialPos(Car["X"], Car["Y"], Car["Z"])
env = DummyVecEnv(
    [
        lambda: Monitor(
            env
        )
    ]
)

# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)
    
def npz_path_to_model(model, path) -> None:
    npy = np.load(path)
    npy = [npy[FileName] for FileName in npy.files]
    params_dict = zip(model.policy.state_dict().keys(), npy)
    state_dict = OrderedDict({k: th.tensor(v) for k, v in params_dict})
    model.policy.load_state_dict(state_dict, strict=True)
    
model_path = 'result_model/2023_Mar_11_Sat_16:30:15_-weights_eta0005.npz'
print("loding evaluation path: ", model_path)
model_name = model_path.split('/')[-1].split('.')[0]
print("loding evaluation model name: ", model_name)

if not os.path.isdir('eval_logs'):
    os.mkdir('eval_logs')

# Initialize RL algorithm type and parameters
model = SAC( #action should be continue
    "CnnPolicy",
    env,
    learning_rate=0.0003,
    verbose=1,
    batch_size=64,
    train_freq=1,
    learning_starts=50, #testing origin 1000
    buffer_size=200000,
    device="auto",
    tensorboard_log="./eval_logs/",
)

#load SAC model(.npz) to model
npz_path_to_model(model, model_path)

# Create an evaluation callback with the same env, called every 10000 iterations
callback_list = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=10000,
    verbose = 1
)
callback_list.append(eval_callback)

# Save a checkpoint every 1000 steps
ep_checkpoint_callback = EpisodeCheckpointCallback(
  check_episodes=1e3,
  save_path="./checkpoint/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
  verbose=2
)
#callback_list.append(ep_checkpoint_callback)

# Stops training when the model reaches the maximum number of episodes
callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=1e4, verbose=1)
#callback_list.append(callback_max_episodes)

callback = CallbackList(callback_list)

# Train for a certain number of timesteps
model.learn(
    total_timesteps=2e4, tb_log_name=f"inter{args.intersection}" + f"_{model_name}_{args.log_name}", callback = callback
)
