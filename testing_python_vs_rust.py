from typing import Any

import my_rust
from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject

from rlgym.utils.obs_builders import ObsBuilder

import numpy as np

from gym import Space
from gym.spaces import Tuple, Box


class TestPythonObsBuilder(ObsBuilder):
    def __init__(self,
                 tick_skip=8,
                 ):
        super().__init__()
        self.tick_skip = tick_skip
        self.index = 0
        self.kickoff_timer = 0
        self.infinite_boost_odds = 0
        self.tick_skip = 4
        # self.rust_obs_builder = my_rust.ObsBuilder(tick_skip = 4, infinite_boost_odds = 0)

    def reset(self, initial_state: GameState):
        self.n = 0
        self.kickoff_timer = 0

    def pre_step(self, state: GameState):
        self.n = 0
        self.kickoff_timer += 1

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray, previous_model_actions: np.ndarray) -> Any:
        obs = [self.kickoff_timer, self.n, player.car_id,
               player.car_data.linear_velocity[0],
               player.car_data.linear_velocity[1],
               player.car_data.linear_velocity[2],
               player.car_data.position[0],
               player.car_data.position[1],
               player.car_data.position[2],
               player.car_data.angular_velocity[0],
               player.car_data.angular_velocity[1],
               player.car_data.angular_velocity[2],
               ]
        self.n += 1
        obs = np.asarray(obs)
        return obs

    def get_obs_space(self) -> Space:
        players = 5
        car_size = 35
        player_size = 10
        return Box(-np.inf, np.inf, (1, 10))
