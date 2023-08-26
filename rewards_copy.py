from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject

from rlgym.utils.reward_functions import RewardFunction

from rlgym.utils.common_values import BLUE_TEAM, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, ORANGE_TEAM, BALL_MAX_SPEED, \
    CAR_MAX_SPEED, BALL_RADIUS, GOAL_HEIGHT, CEILING_Z, BACK_NET_Y, BACK_WALL_Y, SIDE_WALL_X, BOOST_LOCATIONS
from rlgym.utils.math import cosine_similarity


import numpy as np
from numpy.linalg import norm

from typing import Tuple, List

from utils.misc import print_state



class TestReward(RewardFunction):
    # framework for zerosum comes from Nexto code (Rolv and Soren)
    # (https://github.com/Rolv-Arild/Necto/blob/master/training/reward.py)
    def __init__(
            self,
            
    ):
        pass

    def pre_step(self, state: GameState, previous_model_actions: np.ndarray = None):
       pass

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray,
                   previous_model_action: np.ndarray) -> float:
        
        return 1.

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray,
                         previous_model_action: np.ndarray) -> float:
        return 1.