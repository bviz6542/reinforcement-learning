import torch
import gymnasium as gym
from main import Qnet

@torch.no_grad()
def watch_live(model_path: str = "cartpole_dqn.pt", episodes: int = 3) -> None:
    env = gym.make("CartPole-v1", render_mode="human")
    q = Qnet()
    q.load_state_dict(torch.load(model_path, map_location="cpu"))
    q.eval()

    for ep in range(episodes):
        obs, _ = env.reset()
        done, ret = False, 0.0
        while not done:
            # Q-values and greedy action
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            q_vals = q(obs_t)                       # shape (1, 2)
            action = int(q_vals.argmax(dim=1))      # greedy
            # (optional) print decisions
            print(f"Q={q_vals.squeeze().tolist()}, a={action}")

            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            ret += r
        print(f"[watch] episode return: {ret:.1f}")

    env.close()

watch_live("cartpole_dqn.pt", episodes=3)
