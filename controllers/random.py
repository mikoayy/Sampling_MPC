from controllers.interface import Controller


class RandomController(Controller):
    def compute_control(self, obs=None):
        return self.env.action_space.sample()
