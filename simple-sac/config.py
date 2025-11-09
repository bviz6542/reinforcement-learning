from dataclasses import dataclass

@dataclass
class Config:
    env_name: str = "HalfCheetah-v4"
    seed: int = 1
    device: str = "cpu"

    # Replay buffer
    replay_size: int = 100_000
    batch_size: int = 256
    start_steps: int = 1_000

    # SAC
    gamma: float = 0.99
    tau: float = 0.005 # soft target update rate
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    hidden: int = 256
    updates_per_step: int = 1

    # Entropy (temperature)
    auto_alpha: bool = True
    fixed_alpha: float = 0.2

    # Training
    total_steps: int = 100_000
    eval_interval: int = 5_000
    eval_episodes: int = 5

    # Video
    video_interval: int = 20_000