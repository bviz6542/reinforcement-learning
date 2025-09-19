import gymnasium as gym
import collections
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 5e-4
gamma = 0.98
buffer_limit = 5_000
batch_size = 32

class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        # transition: (s, a, r, s_prime, done_mask)
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_list, a_list, r_list, s_prime_list, done_mask_list = [], [], [], [], []
        for (s, a, r, s_prime, done_mask) in mini_batch:
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            s_prime_list.append(s_prime)
            done_mask_list.append(done_mask)

        s      = torch.tensor(np.array(s_list), dtype=torch.float32)              # (B, obs_dim)
        a      = torch.tensor(a_list, dtype=torch.long).unsqueeze(1)              # (B, 1)
        r      = torch.tensor(r_list, dtype=torch.float32).unsqueeze(1)           # (B, 1)
        s_prime= torch.tensor(np.array(s_prime_list), dtype=torch.float32)        # (B, obs_dim)
        done_m = torch.tensor(done_mask_list, dtype=torch.float32).unsqueeze(1)   # (B, 1)
        return s, a, r, s_prime, done_m

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    @torch.no_grad()
    def sample_action(self, observation, epsilon):
        if not torch.is_tensor(observation):
            observation = torch.from_numpy(observation).float()
        q_out = self.forward(observation.unsqueeze(0))   # (1, act_dim)
        if random.random() < epsilon:
            return random.randint(0, 1)                  # CartPole actions
        return int(q_out.argmax(dim=1).item())

def train(q, q_target, memory, optimizer):
    if memory.size() < batch_size:
        return

    for _ in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        # Q(s,a)
        q_out = q(s)                     # (B, act_dim)
        q_a = q_out.gather(1, a)         # (B, 1)

        # target = r + gamma * max_a' Q_target(s', a') * done_mask
        with torch.no_grad():
            max_q_prime = q_target(s_prime).max(dim=1, keepdim=True)[0]  # (B, 1)
            target = r + gamma * max_q_prime * done_mask                 # (B, 1)

        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    env = gym.make('CartPole-v1')
    _, _ = env.reset()

    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())

    memory = ReplayBuffer()
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    print_interval = 10
    score = 0.0

    for n_epi in range(1, 3_001):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))
        obs, info = env.reset()
        done = False

        while not done:
            a = q.sample_action(obs, epsilon)
            next_obs, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            done_mask = 0.0 if done else 1.0
            memory.put((obs, a, r, next_obs, done_mask))

            obs = next_obs
            score += r

        # Train after each episode once buffer is warm
        if memory.size() >= 2_000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0:
            q_target.load_state_dict(q.state_dict())
            avg_score = score / print_interval
            print(f"n_epi: {n_epi}, avg_score: {avg_score:.2f}, n_buffer: {memory.size()}, eps: {epsilon:.3f}")
            score = 0.0

    torch.save(q.state_dict(), "cartpole_dqn.pt")
    print("Saved model -> cartpole_dqn.pt")

    env.close()

if __name__ == '__main__':
    main()
