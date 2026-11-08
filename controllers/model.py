"""Model dla sterowników próbkujących: kopia środowiska z ustawianiem stanu."""
import gymnasium as gym
import numpy as np


def _is_mujoco(env):
    u = env.unwrapped
    return hasattr(u, "data") and hasattr(u, "set_state") and hasattr(u, "model")


def get_state(env):
    """Pełny stan symulatora (nie obserwacja)."""
    u = env.unwrapped
    if _is_mujoco(env):
        return np.concatenate([u.data.qpos, u.data.qvel]).copy()
    return np.array(u.state, dtype=np.float64).copy()


def set_state(env, state):
    """Ustawia pełny stan symulatora i zwraca odpowiadającą mu obserwację."""
    u = env.unwrapped
    state = np.asarray(state, dtype=np.float64)
    if _is_mujoco(env):
        nq = u.model.nq
        u.set_state(state[:nq], state[nq:])
    else:
        u.state = state.copy()
    return u._get_obs()


class EnvModel:
    """Kopia środowiska używana do rolloutów."""

    def __init__(self, env_id, make_kwargs=None, params=None,
                 termination_penalty=0.0, seed=0):
        self.env = gym.make(env_id, **(make_kwargs or {})).unwrapped
        self.env.reset(seed=seed)
        for k, v in (params or {}).items():
            setattr(self.env, k, v)
        self.termination_penalty = termination_penalty

    def get_state(self, env):
        return get_state(env)

    def rollout(self, x0, actions):
        """Skumulowana nagroda sekwencji akcji (T, action_dim) ze stanu x0."""
        set_state(self.env, x0)
        total = 0.0
        horizon = len(actions)
        for t in range(horizon):
            _, reward, terminated, _, _ = self.env.step(actions[t])
            total += reward
            if terminated:
                total -= self.termination_penalty * (horizon - t - 1)
                break
        return total
