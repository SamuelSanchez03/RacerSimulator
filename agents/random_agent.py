from race_sim.agent import Agent
from race_sim.types import Action
import numpy as np
import numpy.typing as npt
import random

class RandomAgent(Agent):
    def __init__(self):
        print("New Random Agent")
        
    def act(self, obs: npt.NDArray[np.float64]) -> Action:
        steer = random.uniform(-1.0, 1.0)
        throttle = random.uniform(0.0, 1.0)
        brake = random.uniform(0.0, 1.0)
        action = Action(steer, throttle, brake)
        print(f"Action: {action}")
        return action
        
    def reset(self, seed=None):
        print("Agent restarted")