"""Funkcje wspólne dla skryptów zadań."""
import gymnasium as gym
import numpy as np

from controllers.model import set_state


def run_episode(env, controller, n_steps, init_state=None, seed=None):
    """Uruchamia jeden epizod i zwraca (skumulowana nagroda, liczba wykonanych kroków)."""
    obs, _ = env.reset(seed=seed)
    if init_state is not None:
        obs = set_state(env, init_state)
    if hasattr(controller, "reset"):
        controller.reset()

    episode_reward = 0.0
    for i in range(n_steps):
        action = np.asarray(controller.compute_control(obs), dtype=np.float64)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        if terminated or truncated:
            break
    return episode_reward, i + 1


def make_pendulum(render=False):
    return gym.make("Pendulum-v1", render_mode="human" if render else None, g=9.81)
