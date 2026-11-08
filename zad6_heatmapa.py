"""Zadanie 6: heatmapa nagrody dla random shooting."""
import itertools
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import make_pendulum, run_episode
from controllers.model import EnvModel
from controllers.random_shooting import RandomShootingController

N_VALUES = [5, 10, 25, 50, 100, 200]
T_VALUES = [5, 10, 20, 30, 50]
N_SEEDS = 3


def evaluate(job):
    n, horizon, seed = job
    env = make_pendulum()
    model = EnvModel("Pendulum-v1", {"g": 9.81})
    controller = RandomShootingController(env, model, n, horizon, seed=seed)
    reward, _ = run_episode(env, controller, 200, init_state=[np.pi, 0.0], seed=seed)
    return job, reward


def plot_heatmap(mean):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mean, cmap="viridis", origin="lower", aspect="auto")
    ax.set_xticks(range(len(T_VALUES)), T_VALUES)
    ax.set_yticks(range(len(N_VALUES)), N_VALUES)
    ax.set_xlabel("długość rolloutu T")
    ax.set_ylabel("liczba rolloutów n")
    ax.set_title(f"Random shooting: skumulowana nagroda epizodu (średnia z {N_SEEDS} ziaren)")
    for i, j in itertools.product(range(len(N_VALUES)), range(len(T_VALUES))):
        ax.text(j, i, f"{mean[i, j]:.0f}", ha="center", va="center",
                color="white" if mean[i, j] < mean.mean() else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="nagroda")
    fig.tight_layout()
    out = "results/heatmap_rs.png"
    fig.savefig(out, dpi=150)
    print(f"Zapisano {out}")


if __name__ == "__main__":
    jobs = list(itertools.product(N_VALUES, T_VALUES, range(N_SEEDS)))
    rewards = np.zeros((len(N_VALUES), len(T_VALUES), N_SEEDS))
    with ProcessPoolExecutor() as pool:
        for (n, t, s), r in pool.map(evaluate, jobs):
            rewards[N_VALUES.index(n), T_VALUES.index(t), s] = r
            print(f"n={n:4d} T={t:3d} ziarno={s}  nagroda={r:9.2f}", flush=True)

    plot_heatmap(rewards.mean(axis=2))
