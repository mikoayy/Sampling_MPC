"""Zadanie 5: random shooting."""
import numpy as np

from common import make_pendulum, run_episode
from controllers.model import EnvModel
from controllers.random_shooting import RandomShootingController

if __name__ == "__main__":
    env = make_pendulum(render=True)
    model = EnvModel("Pendulum-v1", {"g": 9.81})
    controller = RandomShootingController(env, model, n_rollouts=50, horizon=10, seed=0)

    reward, _ = run_episode(env, controller, 200, init_state=[np.pi, 0.0], seed=0)
    print(f"Random shooting (n=50, T=10): nagroda epizodu {reward:.2f}")
    env.close()
