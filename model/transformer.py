import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(len(x), device=x.device).unsqueeze(1)
        x = x + self.positional_embedding(positions).squeeze(1)
        return x


class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerEncoderModel, self).__init__()
        self.model_type = 'TransformerEncoder'
        self.pos_encoder = LearnablePositionalEncoding(ninp)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(input_dim, ninp)  # 입력 차원을 ninp 크기의 임베딩으로 변환
        self.ninp = ninp
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        torch.sqrt(torch.tensor(self.ninp, dtype=torch.float))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output
