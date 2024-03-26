from model.agent import PPONetwork
from AirfoilEnv import CustomAirfoilEnv

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
    env = CustomAirfoilEnv()
    agent = PPONetwork()
    train(env, agent)