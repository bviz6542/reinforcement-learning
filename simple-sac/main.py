import gymnasium as gym
import numpy as np
import torch
from config import Config
from util import set_seed
from replay_buffer import ReplayBuffer
from sac import SAC

def evaluate(env, agent: SAC, episodes=5):
    returns = []
    for _ in range(episodes):
        o, _ = env.reset()
        ep_ret = 0.0
        done = False
        for _ in range(1000):
            a = agent.act(o, deterministic=True)
            step_out = env.step(a)
            o, r, terminated, truncated, _ = step_out
            done = terminated or truncated
            ep_ret += r
            if done:
                break
        returns.append(ep_ret)
    return float(np.mean(returns)), float(np.std(returns))


def main():
    cfg = Config()

    env = gym.make(cfg.env_name)
    test_env = gym.make(cfg.env_name)
    set_seed(env, cfg.seed)
    set_seed(test_env, cfg.seed + 1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    buf = ReplayBuffer(obs_dim, act_dim, cfg.replay_size)
    agent = SAC(obs_dim, act_dim, act_limit, cfg)

    o, _ = env.reset()
    o = np.asarray(o, dtype=np.float32)

    for t in range(1, cfg.total_steps + 1):
        # Collect
        if t < cfg.start_steps:
            a = env.action_space.sample()
        else:
            a = agent.act(o, deterministic=False)

        step_out = env.step(a)
        o2, r, terminated, truncated, info = step_out
        d = terminated or truncated

        buf.add(o, a, r, o2, float(d))
        o = o2 if not d else env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()

        # Update
        if t >= cfg.start_steps:
            for _ in range(cfg.updates_per_step):
                batch = buf.sample(cfg.batch_size)
                info = agent.update(batch, cfg)

        # Log & Eval
        if t % 1000 == 0 and t >= cfg.start_steps:
            print(f"[step {t}] "
                  f"q_loss={info['q_loss']:.3f}  actor_loss={info['actor_loss']:.3f}  "
                  f"alpha={info['alpha']:.3f}  q_pi={info['q_pi']:.2f}")

        if t % cfg.eval_interval == 0:
            mean_ret, std_ret = evaluate(test_env, agent, cfg.eval_episodes)
            print(f"[EVAL step {t}] return={mean_ret:.1f} ± {std_ret:.1f}")

    mean_ret, std_ret = evaluate(test_env, agent, cfg.eval_episodes)
    print(f"FINAL EVAL: return={mean_ret:.1f} ± {std_ret:.1f}")


if __name__ == "__main__":
    main()
