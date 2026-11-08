import numpy as np

from controllers.interface import Controller
from controllers.pd import PDController


class EnergyPDController(Controller):
    """Swing-up przez sterowanie energią, przy górze przełączenie na PD."""

    def __init__(self, env, k_energy=5.0, kp=15.0, kd=5.0, switch_angle=0.4, e_ref=0.0):
        super().__init__(env)
        p = env.unwrapped
        self.m, self.l, self.g = p.m, p.l, p.g
        self.inertia = self.m * self.l ** 2 / 3.0
        self.k_energy = k_energy
        self.switch_angle = switch_angle
        self.e_ref = e_ref
        self.pd = PDController(env, kp=kp, kd=kd, goal=0.0)
        self.mode = "energy"

    def energy(self, theta, theta_dot):
        return 0.5 * self.inertia * theta_dot ** 2 + self.m * self.g * self.l / 2.0 * (np.cos(theta) - 1.0)

    def compute_control(self, obs=None):
        cos_theta, sin_theta, theta_dot = obs
        theta = np.arctan2(sin_theta, cos_theta)  # [-pi, pi], 0 = góra

        if abs(theta) < self.switch_angle:
            self.mode = "pd"
            return self.pd.compute_control(obs)

        self.mode = "energy"
        e = self.energy(theta, theta_dot)
        # w spoczynku sign(0) = 0, więc trzeba ruszyć
        direction = np.sign(theta_dot) if abs(theta_dot) > 1e-3 else 1.0
        action = self.k_energy * (self.e_ref - e) * direction
        return np.clip(np.array([action]), self.env.action_space.low, self.env.action_space.high)
