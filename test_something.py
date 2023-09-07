import rlgym_sim_rs_py
import torch
from torch import set_num_threads
from datetime import datetime
from torch.nn import Linear, Sequential, LeakyReLU
import numpy as np

env = rlgym_sim_rs_py.GymWrapper(tick_skip=8,
                                 team_size=3,
                                 gravity=1.0,
                                 self_play=True,
                                 boost_consumption=1.0,
                                 copy_gamestate_every_step=True,
                                 dodge_deadzone=0.5,
                                 seed=123)

# start timing here
overall_start_time = datetime.now()
iterations = 10
for _ in range(iterations):
    set_num_threads(1)
    actor = Sequential(Linear(231, 256), LeakyReLU(), Linear(256, 256), LeakyReLU(),
                        Linear(256, 256), LeakyReLU(), Linear(256, 90))
    actor.requires_grad_(False)
    done = False
    steps = 0
    start_time = datetime.now()
    for _ in range(10):
        done = False
        obs = env.reset()
        while not done:
            obs = np.array(obs, dtype=np.float32)
            obs = torch.from_numpy(obs)
            actions = actor.forward(obs)
            actions = actions.numpy()
            obs, rewards, done, _ = env.step(actions)
            steps += 1

    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"fps was {steps / elapsed.total_seconds()}")


end_time = datetime.now()
elapsed = end_time - overall_start_time
print(f"completed {iterations} iterations in {elapsed} time")
print(f"total time was {elapsed.total_seconds()}")




