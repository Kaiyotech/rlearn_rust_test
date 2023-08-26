from abc import ABCMeta
from typing import Any

import my_rust
from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject

from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.reward_functions import RewardFunction

import numpy as np

from gym import Space
from gym.spaces import Tuple, Box

# from utils.gamestates import GameState, PlayerData


class TestRustObsBuilder(ObsBuilder):
    def __init__(self,
                 dtap_dict,
                 tick_skip=8,
                 ):
        super().__init__()
        self.tick_skip = tick_skip
        self.index = 0
        self.dtap_dict = dtap_dict
        self.rust_obs_builder = my_rust.ObsBuilder(tick_skip = 4, infinite_boost_odds = 0, dtap_dict = dtap_dict)


    def reset(self, initial_state: GameState):
        self.rust_obs_builder.reset(initial_state)

    def pre_step(self, state: GameState):
        self.rust_obs_builder.pre_step(state)

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray, previous_model_actions: np.ndarray) -> Any:
        self.dtap_dict["hit_towards_bb"] = True
        obs = self.rust_obs_builder.build_obs(state, previous_action)
        return obs

    def get_obs_space(self) -> Space:
        players = 5
        car_size = 35
        player_size = 10
        return Box(-np.inf, np.inf, (1, 10))
    

class testRustRewardsBuilder(RewardFunction):
    def __init__(self) -> None:
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def pre_step(self, state: GameState, previous_model_actions: np.ndarray = None):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, previous_model_action: np.ndarray) -> float:
        return 0.
    
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, previous_model_action: np.ndarray) -> float:
        return 0.
