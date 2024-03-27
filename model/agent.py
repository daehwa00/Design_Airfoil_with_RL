import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class AirfoilCNN(nn.Module):
    def __init__(self):
        super(AirfoilCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 50, 128)  # 입력 길이가 200으로 가정

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 첫 번째 컨볼루션 및 풀링
        x = self.pool(F.relu(self.conv2(x)))  # 두 번째 컨볼루션 및 풀링
        x = x.view(-1, 128 * 50)  # Flatten
        x = F.relu(self.fc1(x))
        return x


class PPONetwork(nn.Module):
    def __init__(self):
        super(PPONetwork, self).__init__()
        self.encoder = AirfoilCNN()
        # policy network
        self.policy = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2), nn.Softmax(dim=-1)
        )
        # value network
        self.value = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self._initialize_weights()

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = self.encoder(x)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
