from rlgym.utils.gamestates import GameState, PlayerData, PhysicsObject
from rlgym.utils.action_parsers import ActionParser
import copy
from typing import Any

import gym.spaces
import numpy as np
from gym.spaces import Discrete

from rlgym.utils import math

from SpectrumObs import CoyoteObsBuilder

class CoyoteAction(ActionParser):
    def __init__(self, version=None):
        super().__init__()
        self._lookup_table = self.make_lookup_table(version)
        # # TODO: remove this
        # self.angle = 0
        # self.counter = 0

    @staticmethod
    def make_lookup_table(version):
        actions = []
        if version is None or version == "Normal":
            # Ground
            for throttle in (-1, 0, 0.5, 1):
                for steer in (-1, -0.5, 0, 0.5, 1):
                    for boost in (0, 1):
                        for handbrake in (0, 1):
                            if boost == 1 and throttle != 1:
                                continue
                            actions.append(
                                [throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
            # Aerial
            for pitch in (-1, -0.75, -0.5, 0, 0.5, 0.75, 1):
                for yaw in (-1, -0.75, -0.5, 0, 0.5, 0.75, 1):
                    for roll in (-1, 0, 1):
                        for jump in (0, 1):
                            for boost in (0, 1):
                                if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                    continue
                                if pitch == roll == jump == 0:  # Duplicate with ground
                                    continue
                                # Enable handbrake for potential wavedashes
                                handbrake = jump == 1 and (
                                        pitch != 0 or yaw != 0 or roll != 0)
                                actions.append(
                                    [boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
            # append stall
            actions.append([0, 1, 0, 0, -1, 1, 0, 0])
            actions = np.array(actions)

        elif version == "flip_reset":
            # Ground
            for throttle in (-1, 0, 1):
                for steer in (-1, 0, 1):
                    for boost in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append(
                            [throttle or boost, steer, 0, steer, 0, 0, boost, 0])
            # Aerial
            for pitch in (-1, 0, 1):
                for yaw in (-1, 0, 1):
                    for roll in (-1, 0, 1):
                        for jump in (0, 1):
                            for boost in (0, 1):
                                if jump == 1 and yaw != 0 or roll != 0 or pitch != 0:  # no flips necessary here
                                    continue
                                if pitch == roll == jump == 0:  # Duplicate with ground
                                    continue
                                # Enable handbrake for potential wavedashes
                                actions.append(
                                    [boost, yaw, pitch, yaw, roll, jump, boost, 0])
            # append stall
            # actions.append([0, 1, 0, 0, -1, 1, 0, 0])
            actions = np.array(actions)

        elif version == "test_dodge":
            # Ground
            for throttle in (-1, 0, 0.5, 1):
                for steer in (-1, -0.5, 0, 0.5, 1):
                    for boost in (0, 1):
                        for handbrake in (0, 1):
                            if boost == 1 and throttle != 1:
                                continue
                            actions.append(
                                [throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
            # Aerial
            for pitch in (-0.75, -0.75, -0.75, -0.75, -0.75, -0.75, -0.75):
                for yaw in (0, 0, 0, 0, 0, 0, 0):
                    for roll in (0, 0, 0):
                        for jump in (0, 1):
                            for boost in (0, 1):
                                if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                    continue
                                if pitch == roll == jump == 0:  # Duplicate with ground
                                    continue
                                # Enable handbrake for potential wavedashes
                                handbrake = jump == 1 and (
                                        pitch != 0 or yaw != 0 or roll != 0)
                                actions.append(
                                    [boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
            # append stall
            actions.append([0, 1, 0, 0, -1, 1, 0, 0])
            actions = np.array(actions)

        elif version == "test_setter":
            # Ground
            for throttle in (-1, 0, 0.5, 1):
                for steer in (-1, -0.5, 0, 0.5, 1):
                    for boost in (0, 1):
                        for handbrake in (0, 1):
                            if boost == 1 and throttle != 1:
                                continue
                            actions.append(
                                [1, 0, 0, 0, 0, 0, 0, 0])
            # Aerial
            for pitch in (-0.85, -0.84, -0.83, 0, 0.83, 0.84, 0.85):
                for yaw in (0, 0, 0, 0, 0, 0, 0):
                    for roll in (0, 0, 0):
                        for jump in (0, 1):
                            for boost in (0, 1):
                                if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                    continue
                                if pitch == roll == jump == 0:  # Duplicate with ground
                                    continue
                                # Enable handbrake for potential wavedashes
                                handbrake = jump == 1 and (
                                        pitch != 0 or yaw != 0 or roll != 0)
                                actions.append(
                                    [1, 0, 0, 0, 0, 0, 0, 0])
            # append stall
            actions.append([1, 0, 0, 0, 0, 0, 0, 0])
            actions = np.array(actions)

        return actions

    def get_action_space(self) -> gym.spaces.Space:
        return Discrete(len(self._lookup_table))

    @staticmethod
    def get_model_action_space() -> int:
        return 1

    def get_model_action_size(self) -> int:
        return len(self._lookup_table)

    def parse_actions(self, actions: Any, state: GameState, zero_boost: bool = False) -> np.ndarray:

        # hacky pass through to allow multiple types of agent actions while still parsing nectos

        # strip out fillers, pass through 8sets, get look up table values, recombine
        parsed_actions = []
        for action in actions:
            # test
            # parsed_actions.append([, 0, 0, 0, 0, 0, 1, 0])
            # continue
            # support reconstruction
            # if action.size != 8:
            #     if action.shape == 0:
            #         action = np.expand_dims(action, axis=0)
            #     # to allow different action spaces, pad out short ones (assume later unpadding in parser)
            #     action = np.pad(action.astype(
            #         'float64'), (0, 8 - action.size), 'constant', constant_values=np.NAN)

            if np.isnan(action).any():  # it's been padded, delete to go back to original
                stripped_action = (
                    action[~np.isnan(action)]).squeeze().astype('int')

                done_action = copy.deepcopy(self._lookup_table[stripped_action])
                if zero_boost:
                    done_action[6] = 0
                parsed_actions.append(done_action)
            elif action.shape[0] == 1:
                action = copy.deepcopy(self._lookup_table[action[0].astype('int')])
                if zero_boost:
                    action[6] = 0
                parsed_actions.append(action)
            else:
                parsed_actions.append(action)
        # # TODO: remove this
        # self.counter += 1
        # if self.counter % 2:
        #     return np.array(
        #         [np.array([0., 0., 0, 0., 0., 0., 0., 0.]), np.array([0., 0., 0, 0., 0., 0., 0., 0.])])
        # else:
        #     return np.array([np.array([0., 0., self.angle, 0., 0., 1., 0., 0.]), np.array([0., 0., self.angle, 0., 0., 1., 0., 0.])])
        return np.asarray(parsed_actions)

if __name__ == '__main__':
    ap = CoyoteAction()
    print(ap.get_action_space())
