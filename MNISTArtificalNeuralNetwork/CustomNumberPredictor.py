"""
This program allows the user to hand draw a digit for the network to predict what number it is. It uses the 
previously trained network to perfrom a forward propagation and obtain a prediction from a pre trained weight and bias file.
"""

import imageReader
import numberPainter
import time
import numpy as np

def f(x):
    return 1 / (1 + np.exp(-x)) #defines the sigmoid normalization function


def forwardPropagate(x,W,B): #a function that performs a forward propagation  on a specified input with sepcified weights and biases
    h = [0,0,0,0]
    z = [0,0,0]

    h[0] = x
    
    for i in range (0,3):
        z[i] = np.dot(W[i],h[i]) + B[i]
        h[i+1] = f(z[i])

    return(h[3],W,B,z,h)

def readPred(predectedVal): #a function that reads what number a network output corresponds by finding the position of the highest entry in the output column vector.
    largestVal = 0
    index = 0
    for i in range (0,10):
        if (predectedVal[i] > largestVal):
            index = i
            largestVal = predectedVal[i]
    return index


while True:
    numberPainter.paintNumber() #opens the code to allow user to paint a custom number
    Xraw = imageReader.convertFile() #converts painted number into a column vector that can be fed into the network

    B = np.load("Bias1.npy", allow_pickle = True) #Loads specified Weights and Biases
    W = np.load("Weights1.npy", allow_pickle = True)

    netOut = forwardPropagate(Xraw,W,B) #performs a forward propagation
    predictedNumber = readPred(netOut[0])
    print(netOut[0])
    print ("Network Predicted Value: " + str(predictedNumber)) #prints prediction
    i = input("Continue (y/n)> ")
    if i == "n":
        break
    else:
        time.sleep(0.5)
        
        
        
        

