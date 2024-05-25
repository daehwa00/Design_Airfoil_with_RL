import torch
import torch.nn as nn
import torch.nn.functional as F


class AirfoilCNN(nn.Module):
    def __init__(self):
        super(AirfoilCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 16, kernel_size=7, stride=2, padding=3
        )  # 340x240 -> 170x120
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=7, stride=2, padding=3
        )  # 170x120 -> 85x60
        self.conv3 = nn.Conv2d(
            32, 64, kernel_size=5, stride=2, padding=2
        )  # 85x60 -> 43x30
        self.conv4 = nn.Conv2d(
            64, 128, kernel_size=5, stride=2, padding=2
        )  # 43x30 -> 22x15
        self.fc1 = nn.Linear(
            128 * 22 * 15, 128
        )  # 최종 컨볼루션 레이어의 출력 크기를 바탕으로 계산된 값
        self.fc2 = nn.Linear(128, 64)  # 추가적인 FC 레이어

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class Actor(nn.Module):
    def __init__(self, n_action):
        super(Actor, self).__init__()
        self.encoder = AirfoilCNN()  # 상태를 인코딩하는 네트워크
        self.policy_mean = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_action)
        )
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
