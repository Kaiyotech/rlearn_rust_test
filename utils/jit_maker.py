import torch
import os
import sys
import inspect

from rocket_learn.agent.discrete_policy import DiscretePolicy
from torch.nn import Linear, Sequential, GELU, LeakyReLU

from agent import Spectrum

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

filename = sys.argv[1]

actor = Sequential(Linear(426, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(), Linear(512, 512),
                   LeakyReLU(),
                   Linear(512, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(),
                   Linear(512, 373))
actor = Spectrum(embedder=Sequential(Linear(35, 128), LeakyReLU(), Linear(128, 35 * 5)), net=actor)

actor = DiscretePolicy(actor, shape=(373,))

# PPO REQUIRES AN ACTOR/CRITIC AGENT
cur_dir = os.path.dirname(os.path.realpath(__file__))
checkpoint = torch.load(os.path.join(cur_dir, filename))
actor.load_state_dict(checkpoint['actor_state_dict'])
actor.eval()
new_name = filename.split("_checkpoint")[0] + "_jit.pt"
test_input_embed = (torch.Tensor(1, 251), torch.Tensor(1, 5, 35))
torch.jit.save(torch.jit.trace(actor, example_inputs=(test_input_embed,)), new_name)

exit(0)
