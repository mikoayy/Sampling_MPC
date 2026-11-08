"""Zadanie 9: MPPI na Hopperze (--balans: model bez nagrody za ruch do przodu)."""
import argparse
import os

import gymnasium as gym

from common import run_episode
from controllers.model import EnvModel
from controllers.mppi import MPPIController

os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--balans", action="store_true",
                        help="model bez nagrody za ruch do przodu: robot tylko balansuje")
    args = parser.parse_args()

    env = gym.make("Hopper-v5", render_mode="human")
    make_kwargs = {"forward_reward_weight": 0.0} if args.balans else {}
    model = EnvModel("Hopper-v5", make_kwargs=make_kwargs, termination_penalty=10.0)
    controller = MPPIController(env, model, n_samples=50, horizon=30, sigma=0.5, lam=1.0, seed=0)

    reward, n_steps = run_episode(env, controller, 300, seed=0)
    x = env.unwrapped.data.qpos[0]
    print(f"Hopper: {n_steps} kroków, nagroda epizodu {reward:.2f}, pozycja x = {x:.2f} m")
    env.close()
