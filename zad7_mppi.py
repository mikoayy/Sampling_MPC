"""Zadanie 7: MPPI."""
import numpy as np

from common import make_pendulum, run_episode
from controllers.model import EnvModel
from controllers.mppi import MPPIController

if __name__ == "__main__":
    env = make_pendulum(render=True)
    model = EnvModel("Pendulum-v1", {"g": 9.81})
    controller = MPPIController(env, model, n_samples=50, horizon=30, sigma=1.0, lam=0.1, seed=0)

    reward, _ = run_episode(env, controller, 200, init_state=[np.pi, 0.0], seed=0)
    print(f"MPPI (n=50, T=30): nagroda epizodu {reward:.2f}")
    env.close()
