"""
This program allows the user to paint their own image and save it locally.
It is used by the predictor trained program to get a user hand drawn image
"""

from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *

def paintNumber(): #defines values for the tkinter paint window
    width = 560
    height = 560
    center = height//2
    white = (255, 255, 255)
    green = (0,128,0)

    def save(): #saves painted image as png file
        filename = "customNumber.png"
        image1.save(filename)
        root.destroy()

    def paint(event): #creates a paint event when canvas on window is clicked, i.e paints wherever window is clicked
        # python_green = "#476042"
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        cv.create_oval(x1, y1, x2, y2, fill="black",width=30)
        draw.line([x1, y1, x2, y2],fill="black",width=30)

    def handler(event): #function to save image when button is pressed
        if (event.char == '\r'):
            save()

    root = Tk() 

    # Tkinter create a canvas to draw on
    cv = Canvas(root, width=width, height=height, bg='white') 
    cv.pack()

    # PIL create an empty image and draw object to draw on
    # memory only, not visible
    image1 = PIL.Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image1)

    # do the Tkinter canvas drawings 
    # cv.create_line([0, center, width, center], fill='green')

    cv.pack(expand=YES, fill=BOTH)
    cv.bind("<B1-Motion>", paint)

    # do the PIL image/draw (in memory) drawings
    # draw.line([0, center, width, center], green)

    # filename = "my_drawing.png"
    # image1.save(filename)
    button=Button(text="save",command=save)
    button.pack()
    root.bind('<Key>',handler)
    root.mainloop()

    
