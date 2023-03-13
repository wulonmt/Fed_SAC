import gym
import airgym
import time

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, StopTrainingOnMaxEpisodes
from stable_baselines3.common.vec_env import VecFrameStack

import utils.setup_path
from utils.Ptime import Ptime
from utils.EpisodeCheckpointCallback import EpisodeCheckpointCallback

import flwr as fl
import numpy as np
from collections import OrderedDict
import torch as th
from torch import nn

import argparse
import json
from subprocess import Popen

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--track", help="which track will be used, 0~2", type=int, default=1)
parser.add_argument("-i", "--intersection", help="which intersection car is in", type=int, default=1)
parser.add_argument("-l", "--log_name", help="modified log name", type=str, nargs='?')
args = parser.parse_args()

with open("settings.json") as f:
    settings = json.load(f)
#settings["ViewMode"] = "SpringArmChase"
settings["ViewMode"] = "NoDisplay"
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

class AirsimClient(fl.client.NumPyClient):
    def __init__(self):
        # Create a DummyVecEnv for main airsim gym env
        self.env = gym.make(
                        "airgym:airsim-car-cont-action-sample-v0",
                        ip_address="127.0.0.1",
                        image_shape=(84, 84, 1),
                    )
        self.env.env.setkwargs(track = args.track)
        self.env.env.setInitialPos(Car["X"], Car["Y"], Car["Z"]) #setting initial pose
        self.env = DummyVecEnv(
            [
                lambda: Monitor(
                    self.env
                )
            ]
        )
        # Frame-stacking with 4 frames
        self.env = VecFrameStack(self.env, n_stack=4)

        # Wrap env as VecTransposeImage to allow SB to handle frame observations
        self.env = VecTransposeImage(self.env)

        # Initialize RL algorithm type and parameters
        self.model = SAC( #action should be continue
            "CnnPolicy",
            self.env,
            learning_rate=0.0003,
            verbose=1,
            batch_size=64,
            train_freq=1,
            learning_starts=50, #testing origin 1000
            buffer_size=200000,
            device="auto",
            tensorboard_log="./tb_logs/",
        )
        
        # Create an evaluation callback with the same env, called every 10000 iterations
        callback_list = []
        eval_callback = EvalCallback(
            self.env,
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
        callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=1e2, verbose=1)
        #callback_list.append(callback_max_episodes)

        self.callback = CallbackList(callback_list)

        self.time = Ptime()
        self.time.set_time_now()
        print("Starting time: ", self.time.get_time())
        self.n_round = int(0)
        
    def get_parameters(self, config):
        policy_state = [value.cpu().numpy() for key, value in self.model.policy.state_dict().items()]
        return policy_state

    def set_parameters(self, parameters):
        params_dict = zip(self.model.policy.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: th.tensor(v) for k, v in params_dict})
        self.model.policy.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.n_round += 1
        self.set_parameters(parameters)
        if("learning_rate" in config.keys()):
            self.model.learning_rate = config["learning_rate"]
        print(f"Training learning rate: {self.model.learning_rate}")
        self.model.learn(
            total_timesteps=1e3,
            tb_log_name=self.time.get_time() + f"inter{args.intersection}" + f"{args.log_name}/SAC_airsim_car_round_{self.n_round}",
            reset_num_timesteps=False,
            callback = self.callback
            )
        return self.get_parameters(config={}), self.model.num_timesteps, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        reward_mean, reward_std = evaluate_policy(self.model, self.env)
        return -reward_mean, self.model.num_timesteps, {"reward mean": reward_mean, "reward std": reward_std} 

def main():        
    # Start Flower client
    fl.client.start_numpy_client(
        server_address="192.168.1.187:8080",
        client=AirsimClient(),
    )
if __name__ == "__main__":
    main()


