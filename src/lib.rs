//use pyo3::prelude::*;

//import stuff yo
use rlgym_sim_rs::{conditionals::common_conditions::TimeoutCondition,
    make,
    obs_builders::{obs_builder::ObsBuilder,
                    advanced_obs::AdvancedObs,
                },               
    reward_functions::{combined_reward::CombinedReward,
                       common_rewards::{ball_goal_rewards,
                       player_ball_rewards},
                       },
    state_setters::random_state::RandomState,
    gym::Gym,
    };
//just for measuring speed
use std::{
    collections::HashMap,
    time::Instant,
};

mod action_parser;

use crate::action_parser::NectoAction;

//use numpy::{PyReadonlyArray, PyArray, Ix1, IntoPyArray};

//use anyhow::Result;
use tch::{Tensor, nn, nn::Module, Device};



// #[pymodule]
// pub fn rlgym_sim_rs_py(_py: Python, m: &PyModule) -> PyResult<()> {
//     m.add_class::<GymWrapper>()?;
//     Ok(())
// }

mod network;

// struct GymReturn{
//     obs: Vec<Vec<f32>>,
//     rewards: Vec<f32>,
//     done: bool,
//     info: HashMap<String, f32>
// }


pub struct GymWrapper {
    gym: Gym,
}

impl GymWrapper {
    #![allow(clippy::too_many_arguments)]
    // #[new]
    /// create the gym wrapper to be used
    // adding a py object here to try and return an array instead for the obs
    pub fn new(team_size: usize, gravity: f32, boost_consumption: f32, tick_skip: usize, _dodge_deadzone: f32, _seed: Option<u64>, self_play: Option<bool>,
        _copy_gamestate_every_step: Option<bool>) -> Self {
        // these need to be set inside Rust because they're Rust
        let parser = Box::new(NectoAction::new());
        //let terminals = Box::new(CombinedTerminalConditions::new(8));
        let terminals = Box::new(TimeoutCondition::new(1500));
        let reward_struct_1 = ball_goal_rewards::VelocityBallToGoalReward::new(Some(false), Some(false));
        let reward_struct_2 = player_ball_rewards::VelocityPlayerToBallReward::new(Some(false));
        let reward_fn = Box::new(CombinedReward::new(vec![Box::new(reward_struct_1), Box::new(reward_struct_2)], vec![1.0, 1.0]));
        let state_set = Box::new(RandomState::new(Some(false), Some(false), Some(false), Some(123)));

        // these come from Python through params
        let spawn_opponents = self_play.unwrap_or(true);
        let _copy_gamestate_every_step = true; //copy_gamestate_every_step.unwrap_or(true);

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

    pub fn reset(&mut self, seed: Option<u64>) -> Vec<Vec<f32>> {
        //TODO need to allow the gamestate in the info hash eventually?
        // TODO figure out a way to return an array directly probably
        self.gym.reset(Some(false), seed)
    }

    // pub fn step<'py>( &mut self, py: Python<'py>, actions: Vec<Vec<f32>>) -> PyResult<(&'py Vec<PyArray<f32, Ix1>>, Vec<f32>, bool, HashMap<String, f32>)> {
    pub fn step( &mut self, actions: Vec<Vec<f32>>) -> (Vec<Vec<f32>>, Vec<f32>, bool, HashMap<String, f32>)  {
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
        // Ok(self.gym.step(actions))
        self.gym.step(actions)
    }

    pub fn step_episode(&mut self, seed: Option<u64>) -> bool {//PyResult<(Vec<&Vec<Vec<f32>>>, Vec<f32>, bool, HashMap<String, f32>)> {
      
        //use a built in rust network for now
        // get first action from the reset
        tch::set_num_threads(1);
        
        let var_store = nn::VarStore::new(Device::Cpu);
        let net = network::net(&var_store.root());
        let start_time = Instant::now();
        let mut steps = 0;
        let mut done = false;
        // let mut all_obs = Vec::with_capacity(160000);
        // let mut all_reward = Vec::with_capacity(160000);
        for _i in 0..10{
            done = false;
            let mut obs = self.gym.reset(Some(false), seed);
            // let mut opt = nn::Adam::default().build(&var_store, 1e-4)?;
            
            while !done{
                // all_obs.push(obs.clone());
                //let tens_obs: Tensor = Tensor::try_from(obs).expect("error from vector to tensor");
                //dbg!(&obs);
                //let tens_obs = vector_of_vectors_to_tensor(&obs);
                let tens_obs = Tensor::from_slice2(&obs);
                let actions: Tensor = tch::no_grad(|| net.forward(&tens_obs));
                let act_vec: Vec<Vec<f32>> = Tensor::try_into(actions).expect("error from tensor to vector");
                let result = self.gym.step(act_vec);
                obs = result.0;
                // all_reward.push(result.1);
                done = result.2;
                steps += 1;
            }
            
        }
        let duration = start_time.elapsed();
        let seconds_elapsed = duration.as_secs_f64();
        println!("{steps} steps in {seconds_elapsed} seconds");
        let fps = steps as f64 / seconds_elapsed;
        println!("fps: {fps}");
        //return (all_obs, reward, done, result.3);
        done
    }
}


// fn vector_of_vectors_to_tensor(data: &Vec<Vec<f32>>) -> Tensor{
//     let rows = data.len() as i64;
//     let cols = data[0].len() as i64;
//     let mut ret_ten = Tensor::empty([rows, cols], kind::FLOAT_CPU);
//     for row in data.iter(){
//         //dbg!(&row);
//         let ten_row = Tensor::from_slice(row);
//         dbg!(&ten_row);
//         ret_ten = Tensor::cat(&[ret_ten.unsqueeze(0), ten_row], 0);
//     }

//     ret_ten
//     // let rows = data.len() as i64;
//     // let cols = data[0].len() as i64;
//     // let flat_data = data.iter().flat_map(|row| row.iter()).cloned().collect::<Vec<_>>();
//     // // Tensor::of_slice(flat_data.as_slice())
//     // // .reshape(&[rows, cols])
//     // // .to_kind(tch::kind::Float)
//     // // Tensor::of_data_size(flat_data.as_slice(), &[rows, cols])
//     // //     .to_kind(kind::Float)
//     // let tensor = Tensor::empty(&[rows, cols], kind::FLOAT_CPU);

//     // // Copy the flattened data into the tensor
//     // tensor.copy_(&Tensor::of_slice(&flat_data));
//     // tensor
// }

