use rlearn_rust_test::GymWrapper;
// use tch::{Tensor, kind, nn, nn::Module, nn::OptimizerConfig, Device, nn::init::Init::Kaiming};

#[test]
fn main() {
    // let var_store = nn::VarStore::new(Device::Cpu);
    // let net = network::net(&var_store.root());
    let mut env = GymWrapper::new(3, 1.0, 1.0, 8, 0.5, Some(123), Some(true), Some(true));
    let result = env.step_episode(Some(123));
    println!("result of step was {result}");
}