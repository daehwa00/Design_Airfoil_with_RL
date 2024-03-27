from model.agent import PPONetwork
from AirfoilEnv import CustomAirfoilEnv, Airfoil
from make_airfoil import get_airfoil_points
from xfoil.xfoil import XFoil
import matplotlib.pyplot as plt

import torch

def train(env, agent, epochs=50, steps_per_epoch=10):
    for epoch in range(epochs):
        state = env.reset()
        total_reward = 0
        for step in range(steps_per_epoch):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                policy, _ = agent(state_tensor)
            action = policy.multinomial(1).item()
            next_state, reward, done, _ = env.step(action)
            agent.optimizer.zero_grad()
            _, value = agent(state_tensor)
            # 여기에 손실 함수와 업데이트 로직을 추가합니다.
            # loss.backward()
            # agent.optimizer.step()
            state = next_state
            total_reward += reward
            if done:
                break
        print(f'Epoch: {epoch}, Total Reward: {total_reward}')

if __name__ == '__main__':
    # env = CustomAirfoilEnv()
    # agent = PPONetwork()
    circles = [((0, 0), 0.005), ((0.1, 0), 0.03), ((0.2, 0), 0.03), ((1, 0), 0.001)]
    airfoil_points = get_airfoil_points(circles,plot=True)
    plt.plot(airfoil_points[:, 0], airfoil_points[:, 1])
    plt.show()
    airfoil = Airfoil(airfoil_points[:, 0], airfoil_points[:, 1])
    xfoil = XFoil()
    xfoil.airfoil = airfoil
    xfoil.Re = 1e6
    xfoil.max_iter = 40
    print(xfoil.a(0))