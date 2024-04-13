from model.agent import Agent
from AirfoilEnv import make_env
from train import Train
from utils import set_seed


ENV_NAME = "CustomAirfoilEnv"
num_points = 80
epochs = 10
T = 20
clip_range = 0.2
beta = 0.01  # 엔트로피 항에 대한 가중치
mini_batch_size = 10
batch_size = 16
processes = batch_size
learning_rate = 1e-4


if __name__ == "__main__":
    set_seed(42)  # 시드 고정
    env = make_env(num_points=num_points)
    agent = Agent(n_actions=2, lr=learning_rate)
    trainer = Train(
        env=env,
        env_name=ENV_NAME,
        agent=agent,
        horizon=T,
        epochs=epochs,
        mini_batch_size=mini_batch_size,
        epsilon=clip_range,
    )
    trainer.step()
