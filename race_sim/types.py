from dataclasses import dataclass
import numpy as np

@dataclass
class Action:
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    
    def __post_init__(self) -> None:
        if not isinstance(self.steer, (int, float)):
            raise TypeError("Invalid action: steer must be a float.")
        if not isinstance(self.throttle, (int, float)):
            raise TypeError("Invalid action: throttle must be a float.")
        if not isinstance(self.brake, (int, float)):
            raise TypeError("Invalid action: brake must be a float.")
        
        self.steer = float(self.steer)
        self.throttle = float(self.throttle)
        self.brake = float(self.brake)
        
        if self.steer < -1.0 or self.steer > 1.0:
            raise ValueError(f'Invalid action: steer must be between -1 and 1.')
        if self.throttle < 0.0 or self.throttle > 1.0:
            raise ValueError(f'Invalid action: throttle must be between 0 and 1.')
        if self.brake < 0.0 or self.brake > 1.0:
            raise ValueError(f'Invalid action: brake must be between 0 and 1.')
        
    def to_numpy(self):
        return np.array([self.steer, self.throttle, self.brake], dtype=np.float64)
        
@dataclass
class CarState:
    x: float
    y: float
    theta: float
    velocity: float = 0.0