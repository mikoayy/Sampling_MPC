"""Zadanie 1-2: stan początkowy wahadła."""
import numpy as np

from common import make_pendulum
from controllers.model import set_state

if __name__ == "__main__":
    env = make_pendulum(render=True)
    env.reset(seed=0)
    obs = set_state(env, [3.14, 0.0])
    print("Obserwacja [cos, sin, theta_dot]:", obs)

    for _ in range(100):
        env.step(np.array([0.0]))
    env.close()
