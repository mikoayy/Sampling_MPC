import numpy as np

from controllers.interface import Controller


class RandomShootingController(Controller):
    """Random shooting: pierwsza akcja najlepszej z losowych sekwencji."""

    def __init__(self, env, model, n_rollouts=50, horizon=20, seed=None):
        super().__init__(env)
        self.model = model
        self.n_rollouts = n_rollouts
        self.horizon = horizon
        self.low = env.action_space.low
        self.high = env.action_space.high
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def compute_control(self, obs=None):
        x0 = self.model.get_state(self.env)
        shape = (self.n_rollouts, self.horizon) + self.low.shape
        sequences = self.rng.uniform(self.low, self.high, size=shape)
        returns = np.array([self.model.rollout(x0, seq) for seq in sequences])
        return sequences[np.argmax(returns), 0]
