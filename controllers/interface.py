from abc import ABC, abstractmethod


class Controller(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def compute_control(self, obs=None):
        """obs = obserwacja z środowiska; sterowniki próbkujące czytają pełny stan z self.env."""
        pass
