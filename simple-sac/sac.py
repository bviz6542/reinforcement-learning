import torch
import numpy as np
import torch.nn.functional as F
from config import Config
from tanh_gaussian_policy import TanhGaussianPolicy
from twinq import TwinQ

class SAC:
    def __init__(self, obs_dim, act_dim, act_limit, cfg: Config):
        self.device = torch.device(cfg.device)
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.act_limit = act_limit

        # Actor & Critics
        self.actor = TanhGaussianPolicy(obs_dim, act_dim, hidden=cfg.hidden).to(self.device)
        self.q = TwinQ(obs_dim, act_dim, hidden=cfg.hidden).to(self.device)
        self.q_target = TwinQ(obs_dim, act_dim, hidden=cfg.hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.critic_lr)

        # Temperature alpha (entropy weight)
        self.auto_alpha = cfg.auto_alpha
        if self.auto_alpha:
            # target entropy = -|A|
            self.target_entropy = -float(act_dim)
            self.log_alpha = torch.tensor(np.log(cfg.fixed_alpha), requires_grad=True, device=self.device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        else:
            self.log_alpha = torch.tensor(np.log(cfg.fixed_alpha), device=self.device)
        self.alpha = self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.actor(obs_t)
        if deterministic:
            # mode of tanh Gaussian = tanh(mu); but we can approximate with mean sample
            a = torch.tanh(dist.base_dist.base_dist.loc)  # (1, act_dim)
        else:
            a = dist.rsample()
        return (a * self.act_limit).cpu().numpy()[0]

    def update(self, batch, cfg: Config):
        obs, act, rew, next_obs, done = (x.to(self.device) for x in batch)
        # scale actions back to [-1,1] domain for critic: our actor already outputs in [-1,1]
        # but env actions are also [-2,2] in Pendulum; we'll store raw env actions.
        # So critics must learn on actual env action scale.
        # (No rescale needed as long as actor & critic see the same scale consistently.)

        # -----------------------
        # Critic update
        # -----------------------
        with torch.no_grad():
            next_dist = self.actor(next_obs)
            next_a = next_dist.rsample()                  # reparameterized
            next_logp = next_dist.log_prob(next_a)        # shape [B]
            next_logp = next_logp.unsqueeze(-1)           # [B,1]

            q1_tp, q2_tp = self.q_target(next_obs, next_a)
            q_tp_min = torch.min(q1_tp, q2_tp)

            alpha = self.log_alpha.exp()
            target_q = rew + (1.0 - done) * cfg.gamma * (q_tp_min - alpha * next_logp)

        q1, q2 = self.q(obs, act)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # -----------------------
        # Actor update
        # -----------------------
        dist = self.actor(obs)
        a = dist.rsample()
        logp = dist.log_prob(a).unsqueeze(-1)  # [B,1]

        q1_pi, q2_pi = self.q(obs, a)
        q_pi = torch.min(q1_pi, q2_pi)

        alpha = self.log_alpha.exp()
        actor_loss = (alpha * logp - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # -----------------------
        # Alpha (temperature) update
        # -----------------------
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.auto_alpha:
            # maximize entropy ⇒ minimize (−α(H + target)) equivalent
            with torch.no_grad():
                # current entropy estimate = -logπ(a|s)
                entropy = -logp  # [B,1]
            alpha_loss = -(self.log_alpha * (entropy + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        # -----------------------
        # Soft target update
        # -----------------------
        with torch.no_grad():
            for p, p_targ in zip(self.q.parameters(), self.q_target.parameters()):
                p_targ.data.mul_(1 - cfg.tau)
                p_targ.data.add_(cfg.tau * p.data)

        return dict(
            q_loss=q_loss.item(),
            actor_loss=actor_loss.item(),
            alpha=float(self.log_alpha.exp().item()),
            alpha_loss=alpha_loss.item() if self.auto_alpha else 0.0,
            q_pi=q_pi.mean().item(),
        )
