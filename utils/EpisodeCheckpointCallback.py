import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback

class EpisodeCheckpointCallback(BaseCallback):
    def __init__(self,
        check_episodes: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0
        ):
        super().__init__(verbose=verbose)
        self.check_episodes = check_episodes
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize
        self.n_episodes = 0
        self.last_check_eps = 0
        
    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.
        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}")
        
    def clean_path(self):
        if len(os.listdir(self.save_path)) > 0:
            print("save path not empty")
            for files in os.listdir(self.save_path):
                os.remove(self.save_path + files)
        else:
            print("save path is already empty")
        
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        self.clean_path()

    def _on_step(self) -> bool:
        # Check that the `dones` local variable is defined
        assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
        self.n_episodes += np.sum(self.locals["dones"]).item()
        to_save = False
        if self.n_episodes % self.check_episodes == 0:
            if self.n_episodes == self.last_check_eps:
                to_save = False  
            else:
                self.last_check_eps = self.n_episodes
                to_save = True

        if self.verbose >= 1 and to_save:
            self.clean_path()
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
                self.model.save_replay_buffer(replay_buffer_path)
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
                self.model.get_vec_normalize_env().save(vec_normalize_path)
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")
        return True
