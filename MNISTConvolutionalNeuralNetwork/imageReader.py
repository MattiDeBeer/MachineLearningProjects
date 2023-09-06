"""
This file opens the custom image saved from the number painter program and converts it into a 28 x 28
matrix for input to the neural networks convolution layer
"""

from PIL import Image, ImageEnhance
import numpy as np

def convertFile(): 
    filename = "customNumber.png" #open image file
    
    im = Image.open(filename) 
    xsize = 28 #sets dimenstion of matrix
    ysize = 28
    im = im.resize((xsize, ysize), Image.BILINEAR) #resizes image to sepcified size
    im = im.convert("L") 
    
    enhancer = ImageEnhance.Contrast(im) #enhances image contrast
    enhanced_im = enhancer.enhance(2.0)

 
    Xraw = np.zeros((28,28)) # makes empty matrix
    
    for y in range(0, ysize): #itterates through high contrast image pixels
        for x in range (0, xsize):
            lum = enhanced_im.getpixel((x,y))  #gets pixe; value
            if (lum > 240): # if pixel is below certian darkness set to -2
                lum = -1
            else: #else scale valu to betwween 1 and 0
                lum = (128 - lum) / 128
            Xraw[y][x] = lum  #append to empty matrix
    return Xraw #return matrix
    
