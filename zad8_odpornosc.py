"""Zadanie 8: odporność MPPI na błąd modelu."""
import itertools
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import make_pendulum, run_episode
from controllers.model import EnvModel
from controllers.mppi import MPPIController

MASS_SCALES = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
SEEDS = range(3)


def evaluate(job):
    scale, seed = job
    env = make_pendulum()
    model = EnvModel("Pendulum-v1", {"g": 9.81}, params={"m": scale})
    controller = MPPIController(env, model, n_samples=50, horizon=30, sigma=1.0, lam=0.1, seed=seed)
    reward, _ = run_episode(env, controller, 200, init_state=[np.pi, 0.0], seed=seed)
    return job, reward


if __name__ == "__main__":
    jobs = list(itertools.product(MASS_SCALES, SEEDS))
    rewards = {scale: [] for scale in MASS_SCALES}
    with ProcessPoolExecutor() as pool:
        for (scale, seed), r in pool.map(evaluate, jobs):
            rewards[scale].append(r)
            print(f"skala={scale:5.2f} ziarno={seed} nagroda={r:9.2f}", flush=True)

    means = np.array([np.mean(rewards[s]) for s in MASS_SCALES])
    stds = np.array([np.std(rewards[s]) for s in MASS_SCALES])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(MASS_SCALES, means, yerr=stds, marker="o", capsize=3)
    ax.set_xlabel("masa w modelu / masa rzeczywista")
    ax.set_ylabel("nagroda epizodu")
    ax.set_title("MPPI: odporność na błąd masy w modelu")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/odpornosc.png", dpi=150)

    print("\nŚrednia ± std nagrody:")
    for scale, m, sd in zip(MASS_SCALES, means, stds):
        print(f"skala={scale:5.2f}  {m:9.2f} ± {sd:6.2f}")
    print("Zapisano results/odpornosc.png")
