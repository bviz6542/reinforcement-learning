# video.py
import os, sys
import gymnasium as gym
import imageio.v2 as imageio

def _pick_backend(explicit: str) -> str:
    if explicit:
        return explicit
    # macOS needs glfw; Linux headless prefers egl; fallback to osmesa
    if sys.platform == "darwin":
        return "glfw"
    if sys.platform.startswith("linux"):
        return os.environ.get("MUJOCO_GL", "egl")
    return "glfw"

def record_episode(agent, env_name: str, out_path: str,
                   steps: int = 1000, deterministic: bool = True,
                   fps: int = 30, seed: int = None,
                   mujoco_gl: str = None):
    """
    Runs one episode with the current policy and saves it as an MP4.
    Works headless if backend supports it.
    """
    backend = _pick_backend(mujoco_gl)
    os.environ["MUJOCO_GL"] = backend   # must be set before env creation

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Use rgb_array so Gymnasium generates frames each render()
    env = gym.make(env_name, render_mode="rgb_array")

    frames = []
    ep_ret = 0.0
    try:
        o, _ = env.reset(seed=seed)
        for _ in range(steps):
            a = agent.act(o, deterministic=deterministic)
            o, r, term, trunc, _ = env.step(a)
            frame = env.render()       # numpy array (H,W,3)
            if frame is not None:
                frames.append(frame)
            ep_ret += float(r)
            if term or trunc:
                break
    finally:
        env.close()

    imageio.mimsave(out_path, frames, fps=fps)
    return ep_ret, len(frames)
