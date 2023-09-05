"""
This program allows the user to paint their own image then save it locally.
It is used by the predictor trained program to get a user hand drawn image
"""

from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *

def paintNumber(): 
    width = 560
    height = 560
    center = height//2
    white = (255, 255, 255)
    green = (0,128,0)

    def save():
        filename = "customNumber.png"
        image1.save(filename)
        root.destroy()

    def paint(event):
        # python_green = "#476042"
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        cv.create_oval(x1, y1, x2, y2, fill="black",width=20)
        draw.line([x1, y1, x2, y2],fill="black",width=20)

    def handler(event):
        if (event.char == '\r'):
            save()

    root = Tk() #creates a Tkinter Window

    # Tkinter create a canvas to draw on
    cv = Canvas(root, width=width, height=height, bg='white') 
    cv.pack()

    # PIL create an empty image and draw object to draw on
    # memory only, not visible
    image1 = PIL.Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image1)

    # do the Tkinter canvas drawings (visible)
    # cv.create_line([0, center, width, center], fill='green')

    cv.pack(expand=YES, fill=BOTH)
    cv.bind("<B1-Motion>", paint)

    # do the PIL image/draw (in memory) drawings
    # draw.line([0, center, width, center], green)

    # PIL image can be saved as .png .jpg .gif or .bmp file (among others)
    # filename = "my_drawing.png"
    # image1.save(filename)
    button=Button(text="Go",command=save)
    button.pack()
    root.bind('<Key>',handler)
    root.mainloop()
    
