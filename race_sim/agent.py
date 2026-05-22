from abc import ABC, abstractmethod
from race_sim.types import Action
import numpy as np
import numpy.typing as npt

class Agent(ABC):
    @abstractmethod
    def act(self, obs: npt.NDArray[np.float64]) -> Action:
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        pass