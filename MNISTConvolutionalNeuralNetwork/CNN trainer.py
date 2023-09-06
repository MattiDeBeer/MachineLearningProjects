"""
This program will train the Convolutional Neural Network to the specified parameters. It created a random weight and 
bias array and then performs a specified amount of backpropagation steps. The parameters can be adjusted in the body of the 
code.
Note that wou will need to download the required dataset files (see ReadMe)
"""

import numpy as np
import time
import math
import random
from scipy import signal
import matplotlib.pyplot as plt

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

def poolLayer(normalizedLayer): #This function defines the algorithm that pools each convolved image (i.e downscales its resolution)
    stride = 3 #defines the scele of the pooling, i.e how many pixels wide the area is that will be combined into one pixel
    dim = len(normalizedLayer[0]) #reading the dimension of the array
    pooledx = [] #defining a new array for each image to be stored in
    
    for p in range (0,4): #itterating through each image
    
        poolArr = np.zeros((int(dim/stride)+ 1 ,int(dim/stride) + 1))#populating the new image with 0's (and giving it the required dimension)
        
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
    
def normalizeLayer(convulutedLayer): #This function removes artefacts around the edges of the of the convolution caused by the sciPy convolution algorithm. For more information see page 63 of the documantation
    for q in range(0,4): #itterate through the array of images
        for j in range(0,31): #itterate through rows in the image
            for k in range(0,31): #itterate through pixels in the row
                 if (convulutedLayer[q][j][k] < 0 or int(convulutedLayer[q][j][k]) == convulutedLayer[q][j][k]): #check for a negative or intager value. If so run the code beneath
                    convulutedLayer[q][j][k] = 0 #turn the current index to a 0.
                    
    return convulutedLayer

def unifyPooledLayers(pooledLayer): #combines all pooled layers into one column vector so it can be fed into the neural network
    unifiedLayer = np.zeros((400,1)) #instantiate an array with 400 elements
    p = 0 #sets the current index of new array
    for i in range(0,4): #itterates through all images of the pooled layer output
        for j in range(0,10): #itterates through all rows in the images
            for k in range(0,10): #itterates through all pixels in the row
                unifiedLayer[p] = pooledLayer[i][j][k] #appends the pixel to the new array
                p += 1 #increments the current index of the new array
                
    return unifiedLayer

def secToTime(secs): #a funtion that figures out the amount of time remaining for training
    hours = math.floor(secs/3600)
    rmins = secs  % 3600
    mins = math.floor(rmins/60)
    secs = int(rmins % 60)
    formattedTime = (str(hours) + ":" + str(mins) + ":" + str(secs))
    return formattedTime
    
            

x = np.load('Xunified.npy',allow_pickle=True) #loads in training data
y = np.load('y1d.npy',allow_pickle=True)
          
filters = [ np.array([[-1,1,1,-1],  #defines an arrayof the filters to be used for the convulutions
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


W0 = np.array(np.random.uniform(-1, 1, (16,400))) #initializes random weight and bias matricies
W1 = np.array(np.random.uniform(-1, 1, (16,16)))
W2 = np.array(np.random.uniform(-1, 1, (10,16)))

B0 = np.array(np.random.uniform(-1, 1,(16,1)))
B1 = np.array(np.random.uniform(-1, 1,(16,1)))
B2 = np.array(np.random.uniform(-1, 1,(10,1)))

h = [0,0,0,0]  #creating the array that holds the outputs to each of the layers in the neural network layers
z = [0,0,0] #this is the output before the normalization was performed

W = [W0,W1,W2] #Pack the rendomised weight and bias matricies into an array
B = [B0,B1,B2]

xAxis = []
yAxis = []

learningRate = 0.001 #sets the learning rate, amount of backpropagation iterations and sample size to be used
itterationNum = 200
setSize = 100

for p in range (0,itterationNum): #itterater over specified itteration number

    startTime = time.time()

    print("-"*40)
    print("Itteration: " + str(p))
    #print("Backpropagating ...")
    costSum = 0

    derivW2Total = 0  #initializes the variables what will sum the derivitives for bakpropagation
    deltal3Total = 0
    derivW1Total = 0
    deltal2Total = 0
    derivW0Total = 0
    deltal1Total = 0

    
        
    sampleStart = random.randint(0,50000 - setSize) #starts a sample in a random position in the dataset
    for j in range(sampleStart , sampleStart + setSize):
        
        convuletedLayer = filterArray(x[j], filters)  #passes the image through the convolution layer
        normalizedLayer = normalizeLayer(convuletedLayer) #removes the artefacts of the convolutin from the edge of the image (see documantation page 63)
        pooledLayer = poolLayer(normalizedLayer) #pools all convolved images (downscales them)
        unifiedLayer = unifyPooledLayers(pooledLayer) #packs all pooled layers into a column vector

        h[0] = unifiedLayer 
    
        for i in range (0,3): #passes the output of the convulution layer into the input of the neural network
            z[i] = np.dot(W[i],h[i]) + B[i]
            h[i+1] = f(z[i])
            
        costVector = calcCostVector(h[3], y[j]) #finds the cost (error) of the prediction
        costSum += sumVector(costVector)
        
        deltal3 = calcDeltal3(h[3],z,y[j]) #calculates the derivitive of the cost w.r.t each weight and bias
        derivW2 = np.dot(deltal3 , np.transpose(h[2]))
            
        deltal2 = calcDeltal2(W,z,deltal3)
        derivW1 = np.dot(deltal2 , np.transpose(h[1]))
            
            
        deltal1 = calcDeltal1(W,z,deltal2)
        derivW0 = np.dot(deltal1 , np.transpose(h[0]))
        
        derivW2Total += derivW2 #appends the derivitives of a single forward propagation to the total
        deltal3Total += deltal3
        derivW1Total += derivW1
        deltal2Total += deltal2
        derivW0Total += derivW0
        deltal1Total += deltal1
    
    W[2] -= learningRate * derivW2Total * (1/setSize) #once the entire sapmle has been put through the network this step updates the weights and biases with the averaged derivitives
    B[2] -= learningRate * deltal3Total * (1/setSize)
    W[1] -= learningRate * derivW1Total * (1/setSize)
    B[1] -= learningRate * deltal2Total * (1/setSize)
    W[0] -= learningRate * derivW0Total * (1/setSize)
    B[0] -= learningRate * deltal1Total * (1/setSize)

    xAxis.append(p)
    yAxis.append(costSum/setSize) #appends the average cost to a plot
    endTime = time.time()
    print("Average cost: " + str(costSum/setSize)) 
    elapsedTime = endTime - startTime #calculates the time for one sample
    print("Time for itteration: "+str(elapsedTime))
    timeLeft = (itterationNum - p) * elapsedTime
    print("Time Left: " + secToTime(timeLeft)) #calculates the time left to run

plt.plot(xAxis,yAxis) #plots the averags cost as a function of sample number
plt.show()


np.save("WeightsCNN1", W) #saves the weights and biases
np.save("BiasCNN1", B)



    
