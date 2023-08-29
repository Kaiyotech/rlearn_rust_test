use anyhow::Result;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, nn::init::Init::Kaiming,
    nn::init::{NormalOrUniform::Normal, FanInOut::{FanIn, FanOut}, NonLinearity::ReLU}};

const HIDDEN_NODES: i64 = 256;
const INPUT_DIM: i64 = 231;
const OUTPUT_DIM: i64 = 90;

pub fn net(my_net: &nn::Path) -> impl Module {
    nn::seq().add(nn::linear(my_net / "layer1",
                INPUT_DIM,
                HIDDEN_NODES,
                nn::LinearConfig{ws_init:Kaiming{dist: Normal,
                    fan: FanIn,
                   non_linearity: ReLU},bs_init:Some(Kaiming{dist: Normal,
                     fan: FanIn,
                    non_linearity: ReLU}),bias:true}))
            .add_fn(|xs| xs.leaky_relu())
            .add(nn::linear(my_net / "layer1",
                HIDDEN_NODES,
                HIDDEN_NODES,
                nn::LinearConfig{ws_init:Kaiming{dist: Normal,
                    fan: FanIn,
                   non_linearity: ReLU},bs_init:Some(Kaiming{dist: Normal,
                     fan: FanIn,
                    non_linearity: ReLU}),bias:true}))
            .add_fn(|xs| xs.leaky_relu())
            .add(nn::linear(my_net / "layer1",
                HIDDEN_NODES,
                HIDDEN_NODES,
                nn::LinearConfig{ws_init:Kaiming{dist: Normal,
                    fan: FanIn,
                   non_linearity: ReLU},bs_init:Some(Kaiming{dist: Normal,
                     fan: FanIn,
                    non_linearity: ReLU}),bias:true}))
            .add_fn(|xs| xs.leaky_relu())
            .add(nn::linear(my_net / "layer1",
                HIDDEN_NODES,
                OUTPUT_DIM,
                nn::LinearConfig{ws_init:Kaiming{dist: Normal,
                    fan: FanIn,
                   non_linearity: ReLU},bs_init:Some(Kaiming{dist: Normal,
                     fan: FanIn,
                    non_linearity: ReLU}),bias:true}))
}