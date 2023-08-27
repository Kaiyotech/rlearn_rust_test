import rlgym_sim_rs_py

env = rlgym_sim_rs_py.GymWrapper(tick_skip=8,
                                      team_size=3,
                                      gravity=1.0,
                                      self_play=True,
                                      boost_consumption=1.0,
                                      copy_gamestate_every_step=True,
                                      dodge_deadzone=0.5,
                                      seed=123)

env.step_episode()
