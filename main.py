import torch
import torch.nn.functional as F
from torch.distributions import Normal
from model.agent import PPONetwork
from AirfoilEnv import CustomAirfoilEnv
from matplotlib import pyplot as plt
from torch import nn
from train import Train
from utils import set_seed

epochs = 50
T = 20
clip_range = 0.2
beta = 0.01  # 엔트로피 항에 대한 가중치
mini_batch_size = 5



if __name__ == "__main__":
    set_seed(42)  # 시드 고정
    env = CustomAirfoilEnv()
    agent = PPONetwork()
    trainer = Train(env=env, env_name="Airfoil", horizon=T, epochs=epochs, mini_batch_size=mini_batch_size, epsilon= clip_range)
    trainer.step()