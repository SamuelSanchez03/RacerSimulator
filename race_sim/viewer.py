
import tkinter as tk
from PIL import ImageTk
from race_sim.types import CarState

class Viewer():
    def __init__(self, track):
        root = tk.Tk()
        track_img = ImageTk.PhotoImage(track.img)

        canvas = tk.Canvas(root, width=track.width, 
        height=track.height)
        root.title(track.path)
        canvas.create_image(0,0, image=track_img, 
        anchor=tk.NW)

        canvas.pack()
        canvas.mainloop()
        
    def render_car(self, state: CarState):
        