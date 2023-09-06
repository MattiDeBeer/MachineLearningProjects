"""
This program tests the trained neural network using the MNIST dataset. It performs the image convolutions with the specified filters
and then a forward propagation on result of the convolution and pooling layers. It then uses the label file to check if the network was 
correct or not. Using this it calculates the accuracy of the network
"""


import numpy as np
import random
from scipy import signal
import time
import matplotlib as plt

def f(x):
    return 1 / (1 + np.exp(-x)) #define the sigmoid normalization function

def f_deriv(x):
    return f(x)*(1-f(x)) #define the derivitive of the sigmoid normalization function

def calcCostVector(predVal, actualVal): #Define a function to calculate the cost vector
    costVector = 0.5*((predVal - actualVal) ** 2)
    return costVector

def calcDeltal3(predVal,z,actualVal): #The below 3 functions calculate the derivitives of the of the cost w.r.t each layerweight and bias
    return (predVal - actualVal) * f_deriv(z[2])

def calcDeltal2(weights,z,deltal3):
    return np.dot(np.transpose(weights[2]), deltal3) * f_deriv(z[1])

def calcDeltal1(weights,z,deltal2):
    return np.dot(np.transpose(weights[1]), deltal2) * f_deriv(z[0])

def sumVector(vector): #defines a function that sums over the enrties in a vector
    total = 0
    for i in range (0, len(vector) -1): 
        total += vector[i][0]
    return total

def poolLayer(normalizedLayer): #inputting the images
    stride = 3 #defines the scele of the pooling, i.e how many pixels wide the area is that will be combined into one pixel
    dim = len(normalizedLayer[0]) #reading the dimension of the array
    pooledx = [] #defining a new array for each image to be stored in
    
    for p in range (0,4): #itterating through each image
    
        poolArr = np.zeros((int(dim/stride)+ 1 ,int(dim/stride) + 1)) #populating the new image with 0's (and giving it the required dimension)
        
        for i in range(0, len(normalizedLayer[p]) - stride, stride): #itterating through the rows in the image, jumping 3  at a time (stride)
            for j in range(0, len(normalizedLayer[p][0]) - stride, stride): #Itterating through each third pixel in the image
                tempH = 0 #setting a temporary variable
                for h in range(0, stride): #itterating through the surrounding pixels to the one currently selected
                    for g in range(0, stride): #itteriating through all pixels in a 3 x 3 grid to the selected one
                        if (normalizedLayer[p][i + h][j + g] > tempH): #checking the temp variable, against the current element
                            tempH = normalizedLayer[p][i + h][j + g] #changing the temp variable if the current element is higher that the temp variable 
                poolArr[int(i/3)][int(j/3)] = tempH #appending the highest local value to the new image array
            
        pooledx.append(poolArr) #appending the pooled image to the new array
    return pooledx #returning the array of pixels

def filterArray(arr, filters): #this function performs the convolutions on the handwritten image using the required filters
    
    xc = [0,0,0,0] #instantiates an empty array (one index per filter)
    for i in range(0,4): #itterate once per filter
        xc[i] = signal.convolve2d(filters[i],arr) #convolves the filter with the image, and populates the array with the output
    
    return(xc)
    
def normalizeLayer(convulutedLayer):
    for q in range(0,4): #itterate through the array of images
        for j in range(0,31): #itterate through rows in the image
            for k in range(0,31): #itterate through pixels in the row
                 if (convulutedLayer[q][j][k] < 0 or int(convulutedLayer[q][j][k]) == convulutedLayer[q][j][k]): #check for a negative or intager value. If so run the code beneath
                    convulutedLayer[q][j][k] = 0 #turn the current index to a 0. 
                    
    return convulutedLayer

def uniformArray(x): #shifts image array values from range (0,255) to (-128,127)
    xTemp = np.zeros((29,28))
    for j in range(0,28):
        for k in range(0,28):
            p = (x[j][k] - 128) / 128
            xTemp[j][k] = p

    return xTemp

def unifyPooledLayers(pooledLayer):
    unifiedLayer = np.zeros((400,1)) #instantiate an array with 400 elements
    p = 0 #sets the current index of new array
    for i in range(0,4): #itterates through all images of the pooled layer output
        for j in range(0,10): #itterates through all rows in the images
            for k in range(0,10): #itterates through all pixels in the row
                unifiedLayer[p] = pooledLayer[i][j][k] #appends the pixel to the new array
                p += 1 #increments the current index of the new array
 
    return unifiedLayer #returns the column vector

def readPred(predectedVal): #This functions trandforms the network output array into a predicted number bu finding the index of the highest entry.
    largestVal = 0
    index = 0
    for i in range (0,10):
        if (predectedVal[i] > largestVal):
            index = i
            largestVal = predectedVal[i]
    return index

def printArray(x): #this function prints out the tested image onto the command line
    for i in range (0,28):
        for j in range(0,28):
            if(x[i][j] > -0.5):
                print("XX", end = "")
            else:
                print("  ", end = "")
        print("")
            

x = np.load('XUnified.npy') #load in the datasets and labels
y = np.load('y2d.npy')
          
filters = [ np.array([[-1,1,1,-1],  #This section defines the filters that will be used in the convolution layer
                      [-1,1,1,-1],
                      [-1,1,1,-1],
                      [-1,1,1,-1]]),
                     
            np.array([[-1,-1,-1,-1],
                      [1,1,1,1],
                      [1,1,1,1],
                      [-1,-1,-1,-1]]),
                     
            np.array([[-1,-1,1,1],
                      [-1,1,1,1],
                      [1,1,1,-1],
                      [1,1,-1,-1]]),

            np.array([[1,1,-1, -1],
                      [1,  1, 1,-1],
                      [-1, 1, 1, 1],
                      [-1,-1, 1, 1]]) ]




h = [0,0,0,0]  #defining the array that holds the outputs to each of the layers in the neural network layers
z = [0,0,0] #this is the output before the normalization was performed

print("Note: If you set the sample size to 1 the program will output an image of the convolution layer")
setSize = int(input("How large would you like your sample size to be (max 49999)> "))

W = np.load("WeightsCNN.npy",allow_pickle=True) #loads in the weights and biases
B = np.load("BiasCNN.npy",allow_pickle=True)

outAns = input("\nwould you like the program to display each network guess and the result? (y/n)> ")

if outAns == "y":
    out = True
else:
    out = False

correctNum = 0
print("Running . . .")
     
sampleStart = random.randint(0,50000 - setSize)  #selects a rendom place in the dataset to start the sample
for j in range(sampleStart , sampleStart + setSize):

    convuletedLayer = filterArray(x[j], filters) #passing the inputs through the convolution layer
    normalizedLayer = normalizeLayer(convuletedLayer) #passing the output through the normalizing layer
    pooledLayer = poolLayer(normalizedLayer) #passing the output through the pooling layer
    unifiedLayer = unifyPooledLayers(pooledLayer) #passing the output through the unifying layer

    h[0] = unifiedLayer #Putting the ourput of the pooled layer into the neural network layers
    
    for i in range (0,3): #performing the foward propagation
        z[i] = np.dot(W[i],h[i]) + B[i]
        h[i+1] = f(z[i])

    predVal = readPred(h[3]) #decoding the networks prediciton from the outpus column vector
    
    if out:
        print('-'*50)  #displaying information about the prediction
        printArray(x[j])
        print("Prediction: " + str(predVal))
        print("Actual: " + str(y[j]))
        if (y[j] == predVal):
            print("correct")
            correctNum += 1
        else:
            print("incorrect")
        
    else:
        if (y[j] == predVal):
            correctNum += 1
        
 
print ("Network accuracy is: " + str((correctNum/setSize)*100) + "%") #averaging and outputting the networks accuracy
 

if (setSize == 1):
    print("Convuluted Layers displayed on plot, the number was " + str(predVal)) #printing an image of the convolution
    #plt.pyplot.imshow(convuletedLayer[0])
    #plt.pyplot.imshow(convuletedLayer[1])
    for i in range(1, 5):
        
        plt.pyplot.subplot(1, 4, i)
        plt.pyplot.imshow(convuletedLayer[i-1])
       # plt.pyplot.imshow(0.5, 0.5, convuletedLayer[i])
            





    
