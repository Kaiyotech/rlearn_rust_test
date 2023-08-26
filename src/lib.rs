use pyo3::prelude::*;

//import stuff yo
use rlgym_sim_rs::{envs::game_match,
    action_parsers::discrete_act::DiscreteAction,
    conditionals::custom_conditions::CombinedTerminalConditions,
    common_values,
    envs,
    gamestates,
    gym,
    make,
    math,
    obs_builders::{obs_builder::ObsBuilder,
                    advanced_obs::AdvancedObs,
                },               
    reward_functions::{combined_reward::CombinedReward,
                       common_rewards::{ball_goal_rewards,
                       player_ball_rewards},
                       },
    state_setters::random_state::RandomState,
    gamestates::game_state::GameState,
    gym::Gym,
    };
//just for measuring speed
use std::{
    collections::HashMap,
    iter::zip,
    path::PathBuf,
    thread::{self, JoinHandle},
    time::Duration,
};

mod action_parser;

use crate::action_parser::NectoAction;

use numpy::{PyReadonlyArray, PyArray, Ix1, IntoPyArray};


#[pymodule]
pub fn rlgym_sim_rs_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<GymWrapper>()?;
    Ok(())
}

//first thing I need
/*            rlgym_sim_rs_py.make(tick_skip=tick_skip, spawn_opponents=spawn_opponents, team_size=team_size, gravity=1.0,
                                 boost_consumption=1.0, copy_gamestate_every_step=True, dodge_deadzone=dodge_deadzone,
                                 terminal_conditions=terminal_conditions, reward_fn=reward_fn, obs_builder=obs_builder,
                                 action_parser=action_parser, state_setter=DynamicGMSetter(state_setter)) */

// also need reset
// and also need step

struct GymReturn{
    obs: Vec<Vec<f32>>,
    rewards: Vec<f32>,
    done: bool,
    info: HashMap<String, f32>
}

#[pyclass(unsendable)]
pub struct GymWrapper {
    gym: Gym,
}

#[pymethods]
impl GymWrapper {
    #![allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (team_size, gravity, boost_consumption, tick_skip, dodge_deadzone, seed=123, self_play=true, copy_gamestate_every_step=true))]
    /// create the gym wrapper to be used
    // adding a py object here to try and return an array instead for the obs
    pub fn new(team_size: usize, gravity: f32, boost_consumption: f32, tick_skip: usize, mut dodge_deadzone: f32, seed: Option<u64>, self_play: Option<bool>,
        copy_gamestate_every_step: Option<bool>) -> Self {
        // these need to be set inside Rust because they're Rust
        let parser = Box::new(NectoAction::new());
        let terminals = Box::new(CombinedTerminalConditions::new(8));
        let reward_struct_1 = ball_goal_rewards::VelocityBallToGoalReward::new(Some(false), Some(false));
        let reward_struct_2 = player_ball_rewards::VelocityPlayerToBallReward::new(Some(false));
        let reward_fn = Box::new(CombinedReward::new(vec![Box::new(reward_struct_1), Box::new(reward_struct_2)], vec![1.0, 1.0]));
        let state_set = Box::new(RandomState::new(Some(false), Some(false), Some(false), Some(123)));

        // these come from Python through params
        let spawn_opponents = self_play.unwrap_or(true);
        let copy_gamestate_every_step = true; //copy_gamestate_every_step.unwrap_or(true);

        // this gets created based on other options but obs lives in Rust
        let mut obs_build_vec = Vec::<Box<dyn ObsBuilder + Send>>::new();
        if spawn_opponents {
            for _ in 0..team_size * 2 {
                obs_build_vec.push(Box::new(AdvancedObs::new()));
            }
        } else {
            for _ in 0..team_size {
                obs_build_vec.push(Box::new(AdvancedObs::new()));
            }
        }
        println!("team size is {team_size}");
        rocketsim_rs::init(None);
        let game_config = make::MakeConfig {
            tick_skip: Some(tick_skip),
            spawn_opponents: Some(spawn_opponents),
            team_size: Some(team_size),
            gravity: Some(gravity),
            boost_consumption: Some(boost_consumption),
            terminal_condition: terminals,
            reward_fn,
            obs_builder: obs_build_vec,
            action_parser: parser,
            state_setter: state_set, 
        };
        let gym = make::make(game_config);
        GymWrapper { gym }
    }

    pub fn reset(&mut self, seed: Option<u64>) -> PyResult<Vec<Vec<f32>>> {
        //TODO need to allow the gamestate in the info hash eventually?
        // TODO figure out a way to return an array directly probably
        Ok(self.gym.reset(Some(false), seed))
    }

    // pub fn step<'py>( &mut self, py: Python<'py>, actions: Vec<Vec<f32>>) -> PyResult<(&'py Vec<PyArray<f32, Ix1>>, Vec<f32>, bool, HashMap<String, f32>)> {
        pub fn step( &mut self, actions: Vec<Vec<f32>>) -> PyResult<(Vec<Vec<f32>>, Vec<f32>, bool, HashMap<String, f32>)> {
        // let mut gym_return= self.gym.step(actions);
        // let mut obs = Vec::<PyArray<f32, Ix1>>::new();
        // for each_obs in gym_return.0{
        //     obs.push(*each_obs.into_pyarray(py))
        // }
        // let result = (&obs, gym_return.1, gym_return.2, gym_return.3);
        // // Ok(match (result){
        // //     Ok() => result,
        // //     Err(e) => panic!("Error returning step from Rust")
        // // })
        // Ok(result)
        Ok(self.gym.step(actions))
    }

}