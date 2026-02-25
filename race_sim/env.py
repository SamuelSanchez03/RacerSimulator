from race_sim.types import CarState, Action
from race_sim.track import Track
from race_sim.viewer import Viewer
import numpy as np
import numpy.typing as npt
import math

W = math.pi/18
FRICTION = 0.95

class RaceEnv: 
    def __init__(self, track_path) -> None:
        self.track = Track(track_path)
        self.state = CarState(275, 450, 0, 0)
        self.viewer = None
        print(f"New env with track '{track_path}'")
        
    def reset(self) -> npt.NDArray[np.float64]:
        self.state = CarState(275, 450, 0, 0)
        print("Env reseted")
        return self.get_obs()
        
    def get_obs(self) -> npt.NDArray[np.float64]:
        x = self.state.x
        y = self.state.y
        theta = self.state.theta
        max_range = 180
        
        obs = []
        for dw in [-math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2]:
            length = self.track.cast_ray(x, y, theta + dw, max_range=max_range)
            obs.append(length/max_range)
        
        print(f"Observation: {obs}")
        return np.array(obs, dtype=np.float64)
    
    def step(self, action: Action) -> tuple[npt.NDArray[np.float64], float, bool]:
        self.state.velocity = FRICTION * (self.state.velocity + action.throttle - action.brake)
        self.state.theta = self.state.theta + action.steer * W 
        
        self.state.x = self.state.x + self.state.velocity * math.cos(self.state.theta)
        self.state.y = self.state.y + self.state.velocity * math.sin(self.state.theta)
        
        done = not self.track.is_on_road(self.state.x, self.state.y)
        
        if done:
            reward = -100.0
        else:
            reward = self.state.velocity
        
        print(f"Step simulated with state: {self.state}\tReward: {reward}")
        return self.get_obs(), reward, done
    
    def render(self) -> None:
        if not self.viewer:
            self.viewer = Viewer(self.track)   
        
        self.viewer.render_car(self.state)    
        self.viewer.update()