import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, act=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), act(),
            nn.Linear(hidden, hidden), act(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)
