import numpy as np

from controllers.interface import Controller


class PDController(Controller):
    """Regulator PD kąta wahadła (theta = 0 to góra)."""

    def __init__(self, env, kp, kd, goal=0.0):
        super().__init__(env)
        self.kp = kp
        self.kd = kd
        self.goal = goal

    def compute_control(self, obs=None):
        cos_theta, sin_theta, theta_dot = obs
        theta = np.arctan2(sin_theta, cos_theta)
        error = self.goal - theta
        action = self.kp * error + self.kd * (-theta_dot)
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action
