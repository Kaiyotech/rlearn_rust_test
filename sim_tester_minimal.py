import rlgym_sim
# from CoyoteParser import CoyoteAction
# from rlgym_tools.extra_state_setters.replay_setter import ReplaySetter
# from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter
# from test_files.test_obs import TestObs
# from test_files.test_reward import TestReward
import numpy as np
from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from testing_rust import TestRustObsBuilder
from testing_python_vs_rust import TestPythonObsBuilder
from testing_rust import testRustRewardsBuilder
from SpectrumParser import CoyoteAction
from datetime import datetime

startTime = datetime.now()

bad_list = []
index = {}
# setter = ReplaySetter("replays/bad_1v1_doubletap_state.npy")
dtap_status = {"hit_towards_bb": False,
                "ball_hit_bb": False,
                "hit_towards_goal": False,
                }
obs = TestRustObsBuilder(dtap_dict=dtap_status)
# obs = TestPythonObsBuilder()
reward = testRustRewardsBuilder()
terminals = [GoalScoredCondition(), TimeoutCondition(1000)]
parser = CoyoteAction()

env = rlgym_sim.make(tick_skip=4, spawn_opponents=True, copy_gamestate_every_step=True,
                     terminal_conditions=terminals, obs_builder=obs, action_parser=parser, team_size=1,
                     reward_fn=reward)

total_steps = 0
num_episodes_to_do = 100
episodes_done = 0
while episodes_done <= num_episodes_to_do:
    done = False
    steps = 0
    env.reset()
    while not done:
        # actions = np.asarray((np.asarray([0]), np.asarray([np.random.randint(0, 373)])))
        # actions = np.asarray(np.asarray([0],))
        # actions = np.asarray([0] * 8), np.asarray([0] * 8)
        actions = np.asarray([np.asarray([1, 0.5, 0.5, 0.5, 0, 0, 1, 0]), np.asarray([1, 0.5, 0.5, 0.5, 0, 0, 1, 0])])
        new_obs, reward, done, state = env.step(actions)
        obs = new_obs
        steps += 1
    total_steps += steps
    episodes_done += 1
    # print(f"completed {steps} steps. Starting new episode. Done {total_steps} total steps")


print(f"executed {total_steps} steps in {datetime.now() - startTime}")

