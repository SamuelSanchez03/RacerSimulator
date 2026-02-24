from PIL import Image
import os
import math

class Track:
    def __init__(self, path):
        self.path = path
        try:
            self.img = Image.open(path).convert("RGB")
            self.pixels = self.img.load()
            self.width, self.height = self.img.size
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The file '{self.path }' was not found. Please check the file path.")
        except IOError as e:
            raise IOError(f"Error: Could not open image file. {e}")
        
    def is_on_road(self, x, y) -> bool:
        x = int(round(x))
        y = int(round(y))
        if x >= self.width or y >= self.height or x < 0 or y < 0:
            return False
    
        return self.pixels[x, y] == (255, 255, 255)
    
    def cast_ray(self, start_x, start_y, theta, max_range=180, step_size=2):
        for length in range(0, max_range+step_size, step_size):
            nx = start_x + length * math.cos(theta)
            ny = start_y + length * math.sin(theta)
            
            if not self.is_on_road(nx, ny):
                return length
            
        return max_range
    
    def __str__(self) -> str:
        return f'Track file: {self.path}\nDimensions: {self.width}x{self.height}'
