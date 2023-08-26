import sys
from redis import Redis
from redis.retry import Retry  # noqa
from redis.backoff import ExponentialBackoff  # noqa
from redis.exceptions import ConnectionError, TimeoutError

# from SpectrumObs import CoyoteObsBuilder
from AdvancedObs import AdvancedObs
from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker
# from rocket_learn.matchmaker.matchmaker import Matchmaker
# from rocket_learn.agent.types import PretrainedAgent
from rocket_learn.utils.truncated_condition import TerminalToTruncatedWrapper
from mybots_terminals import RandomTruncationBallGround
from SpectrumParser import CoyoteAction
from rewards import ZeroSumReward
from rewards_copy import TestReward
from torch import set_num_threads
from setter import CoyoteSetter
from mybots_statesets import EndKickoff
import Constants
import os

# from pretrained_agents.necto.necto_v1 import NectoV1
# from pretrained_agents.nexto.nexto_v2 import NextoV2
# from pretrained_agents.KBB.kbb import KBB

from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition, \
            NoTouchTimeoutCondition

set_num_threads(1)

if __name__ == "__main__":
    # rew = ZeroSumReward(tick_skip=8)
    rew = TestReward()
    frame_skip = Constants.FRAME_SKIP
    fps = 120 // frame_skip
    name = "Default"
    send_gamestate = False
    streamer_mode = False
    local = True
    auto_minimize = True
    game_speed = 100
    evaluation_prob = 0  # 0.02
    past_version_prob = 0  #1  # 0.5  # 0.1
    #non_latest_version_prob = [0.825, 0.0826, 0.0578, 0.0346]  # this includes past_version and pretrained
    non_latest_version_prob = [1, 0, 0, 0]
    deterministic_streamer = True
    force_old_deterministic = True
    gamemode_weights = {'1v1': 0.3, '2v2': 0.7, '3v3': 0}
    visualize = False
    simulator = True
    batch_mode = True
    team_size = 3
    dynamic_game = False
    infinite_boost_odds = 0
    #setter = CoyoteSetter(mode="normal", simulator=False)
    setter = EndKickoff()
    host = "127.0.0.1"
    # model_name = "necto-model-30Y.pt"
    # nectov1 = NectoV1(model_string=model_name, n_players=6)
    # model_name = "nexto-model.pt"
    # nexto = NextoV2(model_string=model_name, n_players=6)
    # model_name = "kbb.pt"
    # kbb = KBB(model_string=model_name)

    # pretrained_agents = Constants_gp.pretrained_agents

    # matchmaker = Matchmaker(sigma_target=0.5, pretrained_agents=pretrained_agents, past_version_prob=past_version_prob,
    #                         full_team_trainings=0.8, full_team_evaluations=1, force_non_latest_orange=False,
    #                         non_latest_version_prob=non_latest_version_prob)

    # terminals = [GoalScoredCondition(),
    #              TerminalToTruncatedWrapper(
    #                  RandomTruncationBallGround(avg_frames_per_mode=[fps * 10, fps * 20, fps * 30],
    #                                             avg_frames=None,
    #                                             min_frames=fps * 10)),
     #            ]
    terminals = [GoalScoredCondition(), TimeoutCondition(fps * 300),]

    if len(sys.argv) > 1:
        host = sys.argv[1]
        if host != "127.0.0.1" and host != "localhost":
            local = False
            batch_mode = True
    if len(sys.argv) > 2:
        name = sys.argv[2]
    # if len(sys.argv) > 3 and not dynamic_game:
    #     team_size = int(sys.argv[3])
    if len(sys.argv) > 3:
        if sys.argv[3] == 'GAMESTATE':
            send_gamestate = True
        elif sys.argv[3] == 'STREAMER':
            streamer_mode = True
            evaluation_prob = 0
            game_speed = 1
            auto_minimize = False
            infinite_boost_odds = 0
            simulator = False
            past_version_prob = 0

            # pretrained_agents = {
            #     nexto: {'prob': 1, 'eval': True, 'p_deterministic_training': 1., 'key': "Nexto"},
            #     kbb: {'prob': 0, 'eval': True, 'p_deterministic_training': 1., 'key': "KBB"}
            # }

            non_latest_version_prob = [1, 0, 0, 0]

            # matchmaker = Matchmaker(sigma_target=1, pretrained_agents=pretrained_agents,
            #                         past_version_prob=past_version_prob,
            #                         full_team_trainings=1, full_team_evaluations=1,
            #                         force_non_latest_orange=streamer_mode,
            #                         non_latest_version_prob=non_latest_version_prob,
            #                         showmatch=True,
            #                         orange_agent_text_file='orange_stream_file.txt'
            #                         )
                                    
            gamemode_weights = {'1v1': 0.3, '2v2': 0.4, '3v3': 0.3}
            
            # terminals = [GoalScoredCondition(), NoTouchTimeoutCondition(fps * 30), TimeoutCondition(fps * 300)]

            # setter = EndKickoff()

        elif sys.argv[3] == 'VISUALIZE':
            visualize = True
            # terminals = [GoalScoredCondition(), NoTouchTimeoutCondition(fps * 30), TimeoutCondition(fps * 10)]
    rust_sim = False
    simulator = False if rust_sim else simulator
    if simulator and not rust_sim:
        from rlgym_sim.envs import Match as Sim_Match
    elif not rust_sim:
        from rlgym.envs import Match
    dtap_dict = {"1", 1}
    flip_reset_dict = {"1", 1}
    if not rust_sim:
        match = Match(
            game_speed=game_speed,
            spawn_opponents=True,
            team_size=team_size,
            state_setter=setter,
            # obs_builder=CoyoteObsBuilder(tick_skip=Constants.FRAME_SKIP, team_size=team_size,
            #                             infinite_boost_odds=infinite_boost_odds, dtap_dict=dtap_dict,
            #                              flip_reset_count_dict=flip_reset_dict),
            obs_builder=AdvancedObs(),
            action_parser=CoyoteAction(),
            terminal_conditions=terminals,
            reward_function=rew,
            tick_skip=frame_skip,
        ) if not simulator else Sim_Match(
            spawn_opponents=True,
            team_size=team_size,
            state_setter=setter,
            # obs_builder=CoyoteObsBuilder(tick_skip=Constants.FRAME_SKIP, team_size=team_size,
            #                             infinite_boost_odds=infinite_boost_odds, dtap_dict=dtap_dict,
            #                              flip_reset_count_dict=flip_reset_dict),
            obs_builder=AdvancedObs(),
            action_parser=CoyoteAction(),
            terminal_conditions=terminals,
            reward_function=rew,
        )
    else:
        match = None
        matchmaker = None

    # local Redis
    if local:
        r = Redis(host=host,
                  username="user1",
                  password=os.environ["redis_user1_key"],
                  db=Constants.DB_NUM,
                  )

    # remote Redis
    else:
        # noinspection PyArgumentList
        r = Redis(host=host,
                  username="user1",
                  password=os.environ["redis_user1_key"],
                  retry_on_error=[ConnectionError, TimeoutError],
                  retry=Retry(ExponentialBackoff(cap=10, base=1), 25),
                  db=Constants.DB_NUM,
                  )


    # pretrained_agents = {nectov1: 0, nexto: 0.05, kbb: 0.05}
    # pretrained_agents = {nexto: PretrainedAgent(prob=0.5, eval=True, p_deterministic_training=1., key="Nexto"),
    #                      kbb: PretrainedAgent(prob=0.5, eval=True, p_deterministic_training=1., key="KBB")}
    # pretrained_agents = None

    worker = RedisRolloutWorker(r, name, match,
                                # matchmaker=matchmaker,
                                sigma_target=1,
                                evaluation_prob=evaluation_prob,
                                force_paging=False,
                                dynamic_gm=dynamic_game,
                                send_obs=True,
                                auto_minimize=auto_minimize,
                                send_gamestates=send_gamestate,
                                gamemode_weights=gamemode_weights,  # default 1/3
                                streamer_mode=streamer_mode,
                                # deterministic_streamer=deterministic_streamer,
                                # force_old_deterministic=force_old_deterministic,
                                # testing
                                batch_mode=batch_mode,
                                step_size=Constants.STEP_SIZE,
                                # pretrained_agents=pretrained_agents,
                                # eval_setter=EndKickoff(),
                                # full_team_evaluations=True,
                                simulator=simulator,
                                # visualize=visualize,
                                live_progress=False,
                                tick_skip=Constants.FRAME_SKIP,
                                # random_boost_states_on_reset=True,
                                rust_sim=rust_sim,
                                team_size=team_size,
                                )

    # worker.env._match._obs_builder.env = worker.env  # noqa
    # if simulator and visualize:
    #     from rocketsimvisualizer import VisualizerThread
    #     arena = worker.env._game.arena  # noqa
    #     v = VisualizerThread(arena, fps=60, tick_rate=120, tick_skip=frame_skip, step_arena=False,  # noqa
    #                          overwrite_controls=False)  # noqa
    #     v.start()

    worker.run()
