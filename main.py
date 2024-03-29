import torch
import torch.nn.functional as F
from torch.distributions import Normal
from model.agent import PPONetwork
from AirfoilEnv import CustomAirfoilEnv
from matplotlib import pyplot as plt
from utils import set_seed

epochs = 50
max_iter = 20
epsilon = 0.2
beta = 0.01  # 엔트로피 항에 대한 가중치


def ppo_update(agent, states, actions, old_log_probs, returns, advantages):
    policy_mean, policy_std, value = agent(states.permute(0, 2, 1))
    dist = Normal(policy_mean, policy_std.exp())
    new_log_probs = dist.log_prob(actions).sum(-1, keepdim=True)

    # Calculate the ratio (pi_theta / pi_theta_old)
    ratio = torch.exp(new_log_probs - old_log_probs)

    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss
    value_loss = F.mse_loss(returns, value)

    # Entropy bonus
    dist_entropy = dist.entropy().mean()

    # Total loss
    loss = policy_loss + 0.5 * value_loss - beta * dist_entropy

    # Take gradient step
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    return policy_loss.item(), value_loss.item(), dist_entropy.item()


def compute_returns_and_advantages(rewards, values, next_value, gamma=0.99, lam=0.95):
    """
    GAE(Generalized Advantage Estimation)로 반환값과 이점을 계산합니다.

    Args:
    - rewards (list of float): 각 타임스텝에서 얻은 보상의 리스트.
    - values (list of float): 에이전트의 가치 함수로부터 얻은 각 타임스텝의 가치 예측값의 리스트.
    - next_value (float): 에피소드 종료 후의 다음 상태의 가치 예측값.
    - gamma (float): 할인 계수.
    - lam (float): GAE 람다 파라미터.

    Returns:
    - returns (torch.Tensor): 각 타임스텝에 대한 반환값.
    - advantages (torch.Tensor): 각 타임스텝에 대한 이점.
    """
    gae = 0
    returns = []
    advantages = []
    # 마지막 타임스텝에서의 다음 가치를 초기값으로 설정
    next_return = next_value

    # 각 타임스텝에 대해 역순으로 계산
    for step in reversed(range(len(rewards))):
        # 해당 타임스텝에서의 반환값 계산
        delta = rewards[step] + gamma * next_return - values[step]
        gae = delta + gamma * lam * gae
        next_return = values[step] + gae
        returns.insert(0, next_return)
        advantages.insert(0, gae)

    # 리스트를 텐서로 변환
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    # 이점을 정규화하여 학습의 안정성을 개선할 수 있음
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

    return returns, advantages


if __name__ == "__main__":
    env = CustomAirfoilEnv()
    agent = PPONetwork()
    episode_returns = []  # 각 에피소드의 총 반환값을 저장하는 리스트
    set_seed(42)  # 시드 고정

    for epoch in range(epochs):
        state = env.reset()
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        episode_return = 0  # 에피소드의 총 반환값 초기화

        for _ in range(max_iter):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).permute(0, 2, 1)
            action_mean, action_std, value = agent(state_tensor)
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            action = action.clamp(0,1)
            log_prob = dist.log_prob(action).sum(-1)

            # 환경으로부터 numpy 배열 형태의 행동이 필요한 경우
            next_state, reward = env.step(action.numpy())

            # 데이터 저장
            states.append(state)
            actions.append(action.numpy())
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value.item())
            episode_return += reward  # 에피소드의 반환값 업데이트

            state = next_state

        episode_returns.append(episode_return)  # 에피소드의 총 반환값 저장

        # 반환값과 이점 계산
        next_value = agent(torch.FloatTensor(next_state).unsqueeze(0).permute(0, 2, 1))[
            -1
        ].item()
        returns, advantages = compute_returns_and_advantages(
            rewards, values, next_value
        )

        # Tensor로 변환
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        log_probs = torch.cat(log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        # PPO 업데이트 수행
        ppo_update(agent, states, actions, log_probs, returns, advantages)

        # 학습이 끝난 후, 에피소드별 총 반환값을 plot
        plt.figure(figsize=(10, 5))
        plt.plot(episode_returns, label="Episode Returns")
        plt.xlabel("Episode")
        plt.ylabel("Total Return")
        plt.title("Episode Returns Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig("episode_returns.png")  # plot을 이미지 파일로 저장
