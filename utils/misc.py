from prettytable import PrettyTable
import numpy as np

from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    critic_params = 0
    actor_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        if "critic" in name:
            critic_params += params
        if "actor" in name:
            actor_params += params
        total_params += params
    print(table)
    print(f"Actor Params: {actor_params}")
    print(f"Critic Params: {critic_params}")
    print(f"Total Trainable Params: {total_params}")
    return total_params


def remove_bad_states():
    files = ["replays/easy_double_tap_1v0.npy", "replays/easy_double_tap_1v1.npy"]
    datas = []
    for file in files:
        datas.append(np.load(file))
    bad_states = [1154, 966]
    bad_states.sort(reverse=True)
    for i, data in enumerate(datas):
        data = np.delete(data, bad_states, 0)
        np.save(files[i], data)


def print_state(state: GameState):
    print("State:")
    print(f"Score is {state.blue_score} - {state.orange_score}")
    print(f"Ball Info:")
    print(f"ang_vel: {state.ball.angular_velocity}")
    print(f"lin_vel: {state.ball.linear_velocity}")
    print(f"position: {state.ball.position}")
    print(f"qaut: {state.ball.quaternion}")
    print(f"Inv Ball Info:")
    print(f"ang_vel: {state.inverted_ball.angular_velocity}")
    print(f"lin_vel: {state.inverted_ball.linear_velocity}")
    print(f"position: {state.inverted_ball.position}")
    print(f"qaut: {state.inverted_ball.quaternion}")
    print(f"Boost_pads: {state.boost_pads}")
    print(f"inv_boost_pads: {state.inverted_boost_pads}")
    print("Player Info:")
    for player in state.players:
        print(player)
        print(f"car_data:")
        print(f"ang_vel: {player.car_data.angular_velocity}")
        print(f"lin_vel: {player.car_data.linear_velocity}")
        print(f"pos: {player.car_data.position}")
        print(f"quat: {player.car_data.quaternion}")
        print(f"inv_car_data:")
        print(f"ang_vel: {player.inverted_car_data.angular_velocity}")
        print(f"lin_vel: {player.inverted_car_data.linear_velocity}")
        print(f"pos: {player.inverted_car_data.position}")
        print(f"quat: {player.inverted_car_data.quaternion}")
    print()
    print()
    print()
