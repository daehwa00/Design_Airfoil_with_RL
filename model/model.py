import torch
import torch.nn as nn
import torch.nn.functional as F

class AirfoilCNN(nn.Module):
    def __init__(self):
        super(AirfoilCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, padding=1)
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=5, padding=1
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 48, 64)  # 입력 길이가 200으로 가정

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 첫 번째 컨볼루션 및 풀링
        x = self.pool(F.relu(self.conv2(x)))  # 두 번째 컨볼루션 및 풀링
        x = x.view(-1, 64 * 48)  # Flatten
        x = F.relu(self.fc1(x))
        return x


class Actor(nn.Module):
    def __init__(self, n_action):
        super(Actor, self).__init__()
        self.encoder = AirfoilCNN()  # 상태를 인코딩하는 네트워크
        self.policy_mean = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_action))
        self.policy_std = nn.Parameter(torch.zeros(n_action))
        self._initialize_weights()

    def forward(self, state):
        x = self.encoder(state)
        action_mean = self.policy_mean(x)
        action_std = torch.exp(self.policy_std).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, action_std)
        return dist

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.policy_std, 0.1)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.encoder = AirfoilCNN()  # 상태를 인코딩하는 네트워크
        self.value = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self._initialize_weights()

    def forward(self, state):
        x = self.encoder(state)
        value = self.value(x)
        return value

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
