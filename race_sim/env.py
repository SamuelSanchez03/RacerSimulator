from race_sim.types import CarState, Action
from race_sim.track import Track
from race_sim.viewer import Viewer
import numpy as np
import numpy.typing as npt
import math

W = math.pi/2
FRICTION = 0.95

class RaceEnv: 
    def __init__(self, track_path, verbose=False) -> None:
        self.track = Track(track_path)
        self.state = self.get_starting_position()
        self.viewer = None
        self.verbose = verbose
        
        self.episode = 0
        self.current_step = 0
        self.current_reward = 0.0
        
        if self.verbose:
            print(f"New env with track '{track_path}'")
            
    def get_starting_position(self) -> CarState:
        x0, y0 = self.track.center_path[0] 
        x1, y1 = self.track.center_path[5] 
        theta = math.atan2(y1 - y0, x1 - x0)
        
        return CarState(x0, y0, theta, 0)
        
    def reset(self) -> npt.NDArray[np.float64]:
        self.state = self.get_starting_position()
        
        self.episode += 1
        self.current_step = 0
        self.current_reward = 0.0
        
        if self.verbose:
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
        
        if self.verbose:
            print(f"Observation: {obs}")
        return np.array(obs, dtype=np.float64)
    
    def step(self, action: Action) -> tuple[npt.NDArray[np.float64], float, bool]:
        self.state.velocity = FRICTION * (self.state.velocity + action.throttle - action.brake)
        self.state.theta = self.state.theta + action.steer * W 
        
        prev_waypoint_idx, _ = self.track.get_closest_waypoint_index(self.state.x, self.state.y)
        self.state.x = self.state.x + self.state.velocity * math.cos(self.state.theta)
        self.state.y = self.state.y + self.state.velocity * math.sin(self.state.theta)
        current_waypoint_idx, current_center_distance = self.track.get_closest_waypoint_index(self.state.x, self.state.y)
        
        done = not self.track.is_on_road(self.state.x, self.state.y)
        
        if done:
            reward = -100000.0
        else:
            reward = self.state.velocity / (current_center_distance + 0.01)
            if prev_waypoint_idx - current_waypoint_idx > 0.8 * len(self.track.center_path):
                reward += 25000.0
            else:
                reward *= (current_waypoint_idx - prev_waypoint_idx)
                
            
        self.current_step += 1
        self.current_reward += reward
        
        if self.verbose:
            print(f"Step simulated with state: {self.state}\tReward: {reward}")
        return self.get_obs(), reward, done
    
    def render(self) -> None:
        if not self.viewer:
            self.viewer = Viewer(self.track)   
        
        self.viewer.render_car(self.state) 
        self.viewer.render_hud(self.episode, self.current_step, self.current_reward)   
        self.viewer.update()