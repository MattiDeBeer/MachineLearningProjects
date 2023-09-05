"""
This Program takes a saved image painted by the user and converts it to a column vector of pixel values.
It is exclusively used by other programs in this project
"""

from PIL import Image
import numpy as np

def convertFile():
    #filename = input("Input the name of the file you want to convert: ") + ".jpg"
    filename = "customNumber.png" #opens custom image file
    
    im = Image.open(filename)
    xsize = 28 #set axis size
    ysize = 28
    im = im.resize((xsize, ysize), Image.BILINEAR) #converts image to specified size
    im = im.convert("L")
        
   
    Xraw = [] #defines an empty array
    
    for y in range(0, ysize): #itterates through image and gets brightness value for each pixel
        for x in range (0, xsize):
            lum = 220 - im.getpixel((x,y))
            if(lum < 0): #If pixel has been drawn on, 'x' is printed (this section diaplays the image)
                lum = 0
                print("  ", end = (""))
            else:
                print("XX", end = (""))
            Xraw.append([lum/200]) #Pixel value is appended to previously created array
        print("")
           
    XCustomRaw = np.array(Xraw) #array with image in is now reformatted to a numpy array, resized and returned
    XCustom = XCustomRaw.reshape(784,1) 
    return XCustom
    #np.save("XCustom", XCustom)
    
