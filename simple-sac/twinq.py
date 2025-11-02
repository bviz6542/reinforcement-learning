import torch
import torch.nn as nn
from mlp import MLP

class TwinQ(nn.Module):
    """Two independent Q networks (for min backup)."""
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)
