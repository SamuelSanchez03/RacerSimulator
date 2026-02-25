from PIL import Image
import math
import numpy as np

class Track:
    def __init__(self, path) -> None:
        self.path = path
        try:
            self.img = Image.open(path).convert("RGB")
            self.width, self.height = self.img.size
            
            img_array = np.array(self.img)
            
            mask = (img_array[:, :, 0] == 255) & (img_array[:, :, 1] == 255) & (img_array[:, :, 2] == 255)
            
            self.grid = mask
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The file '{self.path }' was not found. Please check the file path.")
        except IOError as e:
            raise IOError(f"Error: Could not open image file. {e}")
        
    def is_on_road(self, x: float, y: float) -> bool:
        x = int(round(x))
        y = int(round(y))
        if x >= self.width or y >= self.height or x < 0 or y < 0:
            return False
    
        return self.grid[y, x]
    
    def cast_ray(self, 
                start_x:float, 
                start_y: float, 
                theta: float, 
                max_range: int=180, 
                step_size: int=2) -> float:
        
        distances = np.arange(0, max_range + step_size, step_size)
        nX = start_x + distances * math.cos(theta)
        nY = start_y + distances * math.sin(theta)
        
        nX = np.clip(np.round(nX).astype(int), 0, self.width - 1)
        nY = np.clip(np.round(nY).astype(int), 0, self.height - 1)
        
        on_road_checks = self.grid[nY, nX]
        
        if np.all(on_road_checks):
            return float(max_range)
        else:
            first_collision_idx = np.argmin(on_road_checks)
            return float(distances[first_collision_idx])
    
    def __str__(self) -> str:
        return f'Track file: {self.path}\nDimensions: {self.width}x{self.height}'
