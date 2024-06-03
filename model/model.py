import torch
import torch.nn as nn
import torch.nn.functional as F


class AirfoilCNN(nn.Module):
    def __init__(self):
        super(AirfoilCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 8), stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 7), stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 7), stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 8), stride=2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 4), stride=2)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(2, 3), stride=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) 
        return x


class Actor(nn.Module):
    def __init__(self, n_action):
        super(Actor, self).__init__()
        self.encoder = AirfoilCNN()  # 상태를 인코딩하는 네트워크
        self.policy_mean = nn.Sequential(
            nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, n_action)
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
        self.value = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 1))
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
