"""
This Program allows you to view a given entry in the MNIST dataset. Just input the index you wan to view.
"""


import numpy as np

x = np.load("X.npy") #load the MNIST dataset images and labels
y = np.load("Y.npy")

while True:
    
    n = int(input("Index you want to view (max 49999): ")) #gets user input
    #print(x[n]) 
    p = 0 
    for i in range(0,27): #itterates through the mnist matrix and displays it as a text image
        print(" ")
        for j in range(0,28):
            temp = (x[n][p][0])
            if (temp == 0):
                print("  ", end=(''))
            else:
                print("XX",end=(''))
            p += 1
           
    tempHighestY = 0 #itterates through the label matrix to find what the number is 
    for k in range (0,9):
        if (y[n][k] > tempHighestY):
            tempHighest = y[i]
            actualNum = k

    print("")
    print("This entry is a: " + str(actualNum)) #outputs the label
    
    cont = input("would you like to view another index? (y/n)> ") #asks the user if they want to view another image
    if cont == "n":
        break

