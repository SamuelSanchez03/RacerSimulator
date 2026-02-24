
import tkinter as tk
from PIL import ImageTk
from race_sim.types import CarState
import math

CAR_WIDTH = 22
CAR_LENGTH = 14

class Viewer():
    def __init__(self, track):
        self.root = tk.Tk()
        self.track = track
        self.track_img = ImageTk.PhotoImage(track.img)

        self.canvas = tk.Canvas(self.root, width=track.width, height=track.height)
        self.root.title(track.path)
        self.canvas.create_image(0,0, image=self.track_img, anchor=tk.NW)

        self.canvas.pack()
        
        car_state = CarState(275, 450, -math.pi/8, 0) 
        self.render_car(car_state)
        
        self.root.mainloop()
        
    #def render_rays(self, state: CarState):
        
    def render_car(self, state: CarState):
        x = state.x
        y = state.y
        velocity = state.velocity
        theta = state.theta
        dx = CAR_WIDTH / 2
        dy = CAR_LENGTH / 2
        
        c = math.cos(theta)
        s = math.sin(theta)
        
        x1, y1 = x + (dx * c - (-dy) * s), y + (dx * s + (-dy) * c)
        x2, y2 = x + (dx * c - dy * s), y + (dx * s + dy * c)
        x3, y3 = x + ((-dx) * c - dy * s), y + ((-dx) * s + dy * c)
        x4, y4 = x + ((-dx) * c - (-dy) * s), y + ((-dx) * s + (-dy) * c)
        
        points = [
            x1, y1,
            x2, y2,
            x3, y3,
            x4, y4
        ]

        self.canvas.create_polygon(points, fill='red', outline='')
        self.canvas.create_line(x, y, x + CAR_WIDTH/2 * c, y + CAR_WIDTH/2 * s, fill='green', arrow=tk.LAST)
        
        for dw in [-math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2]:
            length = self.track.cast_ray(x, y, theta + dw)
            self.canvas.create_line(x, y, x + length * math.cos(theta + dw), y + length * math.sin(theta + dw), fill='blue')