from typing import Tuple
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size: int):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.max_size, self.ptr, self.size = size, 0, 0

    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = (
            torch.as_tensor(self.obs_buf[idxs]),
            torch.as_tensor(self.acts_buf[idxs]),
            torch.as_tensor(self.rews_buf[idxs]),
            torch.as_tensor(self.next_obs_buf[idxs]),
            torch.as_tensor(self.done_buf[idxs]),
        )
        return batch
