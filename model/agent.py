import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PPONetwork(nn.Module):
    def __init__(
        self, input_dim, ninp, nhead, nhid, nlayers, dropout=0.5
    ):  # 필요한 파라미터들
        super(PPONetwork, self).__init__()
        self.encoder = AirfoilCNN(
            input_dim, ninp, nhead, nhid, nlayers, dropout
        )  # 앞서 정의한 TransformerModel 사용
        # policy network
        self.policy = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 3), nn.Softmax(dim=-1)
        )
        # value network
        self.value = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = self.encoder(x)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value


class AirfoilCNN(nn.Module):
    def __init__(self):
        super(AirfoilCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(
            128 * 50, 100
        )  # 차원 조정이 필요함. 200 -> 100 (MaxPooling) -> 50 (두 번째 MaxPooling)
        self.fc2 = nn.Linear(100, 1)  # 최종 출력, 예를 들어 양력 계수

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 첫 번째 컨볼루션 및 풀링
        x = self.pool(F.relu(self.conv2(x)))  # 두 번째 컨볼루션 및 풀링
        x = x.view(-1, 128 * 50)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
