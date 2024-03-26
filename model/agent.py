from .transformer import TransformerEncoderModel

import torch.nn as nn
import torch.optim as optim


class PPONetwork(nn.Module):
    def __init__(self, input_dim, ninp, nhead, nhid, nlayers, dropout=0.5):  # 필요한 파라미터들
        super(PPONetwork, self).__init__()
        self.encoder = TransformerEncoderModel(input_dim, ninp, nhead, nhid, nlayers, dropout)  # 앞서 정의한 TransformerModel 사용
        # policy network
        self.policy = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Softmax(dim=-1)
        )
        # value network
        self.value = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = self.encoder(x)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value
    