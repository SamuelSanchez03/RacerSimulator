
import tkinter as tk
from PIL import ImageTk
from race_sim.types import CarState
from race_sim.track import Track
import math

CAR_WIDTH = 14
CAR_LENGTH = 22

class Viewer():
    def __init__(self, track: Track) -> None:
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.track = track
        self.track_img = ImageTk.PhotoImage(self.track.img)

        self.canvas = tk.Canvas(self.root, width=self.track.width, height=self.track.height)
        self.root.title(self.track.path)
        self.canvas.create_image(0,0, image=self.track_img, anchor=tk.NW)
        self.render_waypoints()

        self.canvas.pack()
        self.is_open = True
        
    def on_close(self) -> None:
        self.is_open = False
        self.root.destroy()

    def update(self) -> None:
        self.root.update()
        
    def render_waypoints(self):
        for x, y in self.track.center_path:
            self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill='yellow', outline='')

    def render_car(self, state: CarState) -> None:
        x = state.x
        y = state.y
        theta = state.theta
        dx = CAR_LENGTH / 2
        dy = CAR_WIDTH / 2
        
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
        
        self.canvas.delete('car')

        self.canvas.create_polygon(points, fill='red', outline='', tags='car')
        self.canvas.create_line(x, y, x + CAR_WIDTH/2 * c, y + CAR_WIDTH/2 * s, fill='green', arrow=tk.LAST, tags='car')
        
        for dw in [-math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2]:
            length = self.track.cast_ray(x, y, theta + dw)
            self.canvas.create_line(x, y, x + length * math.cos(theta + dw), y + length * math.sin(theta + dw), fill='blue', tags='car')
            
    def render_hud(self, episode: int, step: int, reward: float) -> None:
        # Borramos el texto anterior usando su etiqueta
        self.canvas.delete('hud')
        
        # Preparamos el texto a mostrar
        hud_text = f"Episode: {episode}\nStep: {step}\nReward: {reward:.2f}"
        
        # Dibujamos el texto en las coordenadas (10, 10) alineado a la izquierda (NW)
        self.canvas.create_text(
            10, 10, 
            text=hud_text, 
            anchor=tk.NW, 
            fill="white", # Puedes cambiar el color si tu pista es muy oscura
            font=("Arial", 14, "bold"), 
            tags='hud'
        )