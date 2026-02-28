from abc import ABC, abstractmethod
from race_sim.types import Action
import numpy as np
import numpy.typing as npt

class Agent(ABC):
    @abstractmethod
    def act(self, obs: npt.NDArray[np.float64]) -> Action:
        pass
    
    @abstractmethod
    def reset(self, seed=None):
        pass