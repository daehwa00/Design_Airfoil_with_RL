import torch
import torch.nn as nn
from torch.optim import Adam
from model.model import Actor, Critic


class Agent(nn.Module):
    def __init__(self, n_actions=2, lr=1e-4):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Actor
        self.actor = Actor(n_actions).to(self.device)
        # Critic
        self.critic = Critic().to(self.device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr, eps=1e-8)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr * 5, eps=1e-8)

        self.critic_loss = torch.nn.MSELoss()

    def optimize(self, actor_loss, critic_loss):
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def choose_dists(self, state, use_grad=True):
        if use_grad:
            dist = self.actor(state)
        else:
            with torch.no_grad():
                dist = self.actor(state)
        return dist

    def get_value(self, state, use_grad=True):
        # concat
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if use_grad:
            value = self.critic(state)

        else:
            with torch.no_grad():
                value = self.critic(state)
        return value

    def choose_actions(self, dist):
        action = dist.sample()
        return action

    def scale_actions(self, actions):
        actions = nn.Sigmoid()(actions)
        scaled_actions = torch.zeros_like(actions)
        a = 0.1
        # x 값을 0 ~ 0.8로 스케일링
        scaled_actions[0][0] = actions[0][0] * (1 - a)
        # r 값을 0~0.2로 스케일링
        scaled_actions[0][1] = actions[0][1] * a

        return scaled_actions
