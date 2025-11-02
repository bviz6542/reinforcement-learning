import torch
import torch.nn as nn
from torch.distributions import Normal, TransformedDistribution, Independent
from torch.distributions.transforms import TanhTransform
from mlp import MLP

class TanhGaussianPolicy(nn.Module):
    """
    Actor: outputs a Tanh-squashed Gaussian.
    Returns a torch Distribution that supports .rsample() and .log_prob().
    """
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = MLP(obs_dim, 2 * act_dim, hidden)
        self.act_dim = act_dim

    def forward(self, obs) -> TransformedDistribution:
        mu_logstd = self.net(obs)
        mu, log_std = mu_logstd[..., :self.act_dim], mu_logstd[..., self.act_dim:]
        log_std = torch.tanh(log_std)
        min_log, max_log = (-20, 2)
        log_std = min_log + 0.5 * (log_std + 1.0) * (max_log - min_log)
        std = torch.exp(log_std)

        base = Independent(Normal(loc=mu, scale=std), 1)
        dist = TransformedDistribution(base, TanhTransform(cache_size=1))
        return dist
