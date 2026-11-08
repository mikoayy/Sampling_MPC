import numpy as np

from controllers.interface import Controller


class MPPIController(Controller):
    """MPPI (Algorytm 1 z instrukcji)."""

    def __init__(self, env, model, n_samples=100, horizon=30, sigma=1.0, lam=1.0,
                 n_iters=1, seed=None):
        super().__init__(env)
        self.model = model
        self.n_samples = n_samples
        self.horizon = horizon
        self.low = env.action_space.low
        self.high = env.action_space.high
        self.sigma = np.broadcast_to(np.asarray(sigma, dtype=np.float64), self.low.shape)
        self.lam = lam
        self.n_iters = n_iters
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.u = np.zeros((self.horizon,) + self.low.shape)

    def compute_control(self, obs=None):
        x0 = self.model.get_state(self.env)
        for _ in range(self.n_iters):
            noise = self.rng.normal(0.0, 1.0, (self.n_samples,) + self.u.shape) * self.sigma
            u_n = np.clip(self.u + noise, self.low, self.high)        # clamp do zakresu
            delta = u_n - self.u                                        # efektywne zaburzenia
            costs = np.array([-self.model.rollout(x0, seq) for seq in u_n])
            costs -= costs.min()                                        # stabilność numeryczna exp
            w = np.exp(-costs / self.lam)
            w /= w.sum()
            self.u = self.u + np.tensordot(w, delta, axes=1)

        action = self.u[0].copy()
        self.u = np.concatenate([self.u[1:], np.zeros_like(self.u[:1])])  # przesunięcie w czasie
        return action
