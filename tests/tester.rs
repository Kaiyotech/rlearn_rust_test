use rlearn_rust_test::GymWrapper;

#[test]
fn main() {
    let env = GymWrapper::new(3, 1.0, 1.0, 8, 0.5, 123, Some(true), Some(true));
    env.step_episode(123);
}