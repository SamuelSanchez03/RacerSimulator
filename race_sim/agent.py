from abc import ABC, abstractmethod
from race_sim.types import Action


class Agent(ABC):
    @abstractmethod
    def act(self, obs) -> Action:
        pass
    @abstractmethod
    def reset(self, seed=None):
        pass