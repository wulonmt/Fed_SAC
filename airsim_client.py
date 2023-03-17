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
from utils.FedSAC import FedSAC

import flwr as fl
import numpy as np
from collections import OrderedDict
import torch as th
from torch import nn

import argparse
import json
from subprocess import Popen
import random

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
    def __init__(self, Fed_target = True, shuffle_Q = False ):
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
        self.model = FedSAC( #action should be continue
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
            shuffle_doubleQ = suffle_Q
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
        self.Fed_target = Fed_target
        
    def swap_Q(self):
        qf0_keys, qf1_keys = [], []
        for key in self.model.policy.state_dict().keys():
            if "critic.qf0" in key:
                qf0_keys.append(key)
            elif "critic.qf1" in key:
                qf1_keys.append(key)
                
        qf0 = [self.model.policy.state_dict()[key] for key in qf0_keys]
        qf1 = [self.model.policy.state_dict()[key] for key in qf1_keys]
        qf0, qf1 = qf1, qf0
        qf0_pair = zip(qf0_keys, qf0)
        qf1_pair = zip(qf1_keys, qf1)
        qf0_dict = {key : value.clone().detach() for key, value in qf0_pair}
        qf1_dict = {key : value.clone().detach() for key, value in qf1_pair}

        self.model.policy.load_state_dict(qf1_dict, strict=False)
        self.model.policy.load_state_dict(qf0_dict, strict=False)
        self.model.doubleQ_swapped = not self.model.doubleQ_swapped
        
    def set_Fed_target(self):
        target_keys, critic_keys = [], []
        for key in self.model.policy.state_dict().keys():
            if "target" in key:
                target_keys.append(key)
            elif "critic" in key:
                critic_keys.append(key)
        critic_value = [self.model.policy.state_dict()[key] for key in critic_keys]
        target_pair = zip(target_keys, critic_value)
        target_dict = {key : value.clone().detach() for key, value in target_pair}
        self.model.policy.load_state_dict(target_dict, strict=False)
        
    def get_parameters(self, config):
        policy_state = [value.cpu().numpy() for key, value in self.model.policy.state_dict().items()]
        return policy_state

    def set_parameters(self, parameters):
        params_dict = zip(self.model.policy.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: th.tensor(v) for k, v in params_dict})
        self.model.policy.load_state_dict(state_dict, strict=True)
        if random.random() > 0.5:
            self.swap_Q()
        if self.Fed_target:
            self.set_Fed_target()

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


