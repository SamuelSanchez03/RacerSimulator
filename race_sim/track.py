from PIL import Image
import math
import numpy as np
from skimage.morphology import skeletonize

class Track:
    def __init__(self, path) -> None:
        self.path = path
        try:
            self.img = Image.open(path).convert("RGB")
            self.width, self.height = self.img.size
            
            img_array = np.array(self.img)
            
            mask = (img_array[:, :, 0] == 255) & (img_array[:, :, 1] == 255) & (img_array[:, :, 2] == 255)
            
            self.grid = mask
            self.center_line_grid = self.extract_center_line()
            
            y_coords, x_coords = np.where(self.center_line_grid)
            center_points = list(zip(x_coords, y_coords))
            self.center_path = self.sort_center_points(center_points)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The file '{self.path }' was not found. Please check the file path.")
        except IOError as e:
            raise IOError(f"Error: Could not open image file. {e}")
        
    def extract_center_line(self) -> np.ndarray:
        return skeletonize(self.grid)
    
    def distance(self, point1: tuple[int, int], point2: tuple[int, int]) -> int:
        return (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2
    
    def sort_center_points(self, center_points: list) -> list:
        ordered_points = [center_points.pop(0)]
        
        while center_points:
            min_distance = 100000.0
            closest_idx = 0
            for i, point in enumerate(center_points):
                current_distance = self.distance(point, ordered_points[-1])
                if current_distance < min_distance:
                    closest_idx = i
                    min_distance = current_distance
                
            ordered_points.append(center_points.pop(closest_idx))
            
        return ordered_points
    
    def get_closest_waypoint_index(self, x: float, y: float) -> tuple[int, float]:
        min_distance = 100000.0
        closest_idx = 0
        for i, waypoint in enumerate(self.center_path):
            current_distance = self.distance((int(x), int(y)), waypoint)
            if current_distance < min_distance:
                closest_idx = i
                min_distance = current_distance
                
        return closest_idx, min_distance
        
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
