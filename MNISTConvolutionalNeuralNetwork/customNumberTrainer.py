"""
This program prompts the user to hand draw a digit. It then runs this digit through the CNN and predicts what it is. If incorrect it will perform a 
backpropagation step in order to home on its accuracy.
"""

import numpy as np
import imageReader
from scipy import signal
import numberPainter
import time

def f(x):
    return 1 / (1 + np.exp(-x)) #define the sigmoid normalization function

def f_deriv(x):
    return f(x)*(1-f(x)) #define the derivitive of the sigmoid normalization function

def calcCostVector(predVal, actualVal):  #Define a function to calculate the cost vector
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

def readPred(predectedVal):  #function reads the prediction of the network by finding the index with the highest entry
    largestVal = 0
    index = 0
    for i in range (0,10):
        if (predectedVal[i] > largestVal):
            index = i
            largestVal = predectedVal[i]
    return index

def printArray(x): #formats an image from the dataset and prints it to the command line
    for i in range (0,28):
        for j in range(0,28):
            if(x[i][j] > -0.5):
                print("XX", end = "")
            else:
                print("  ", end = "")
        print("")

def generateNumberVector(value): #generates a vector from a number
    vector = np.zeros((10,1)) #makes a column vector full of zeros
    vector[value][0] = 1.0 #puts a 1 in the index of the argument
    return vector
            

x = np.load('XUnified.npy') #loads training data and labels
y = np.load('y2d.npy')

learningRate = 0.005 #sets learning rate 
          
filters = [ np.array([[-1,1,1,-1], #defines an arrayof the filters to be used for the convulutions
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




h = [0,0,0,0] #creating the array that holds the outputs to each of the layers in the neural network layers
z = [0,0,0] #this is the output before the normalization was performed
 

W = np.load("WeightsCNN.npy",allow_pickle=True) #loads weight and bias matricies
B = np.load("BiasCNN.npy",allow_pickle=True)

correctNum = 0
     
while True:
    numberPainter.paintNumber() #opens number painter window
    x = imageReader.convertFile() #converts file into image
    printArray(x) #prints image to command  line

    convuletedLayer = filterArray(x, filters) #passes the image through the convolution layer
    normalizedLayer = normalizeLayer(convuletedLayer) #removes the artefacts of the convolutin from the edge of the image (see documantation page 63)
    pooledLayer = poolLayer(normalizedLayer) #pools all convolved images (downscales them)
    unifiedLayer = unifyPooledLayers(pooledLayer) #packs all pooled layers into a column vector

    h[0] = unifiedLayer
    #printArray(x[j])

    for i in range (0,3): #passes the output of the convulution layer into the input of the neural network
        z[i] = np.dot(W[i],h[i]) + B[i]
        h[i+1] = f(z[i])

    predVal = readPred(h[3]) #finds teh value the network predicts
    print("Predicted Value: " + str(predVal)) #outputs this prediction
    correct = input("Was this prediction Correct? (y/n): ") #asks the user if the prediciton is correct
    if (correct == "n"): 
        try:
            actualVal = int(input("What was the actual value?: ")) 
            y = generateNumberVector(actualVal) #generates the correct vector for the prediction

            deltal3 = calcDeltal3(h[3],z,y) #calculates the derivitive of the cost function w.t.r each weight and bias
            derivW2 = np.dot(deltal3 , np.transpose(h[2])) 
                
            deltal2 = calcDeltal2(W,z,deltal3)
            derivW1 = np.dot(deltal2 , np.transpose(h[1]))
                
            deltal1 = calcDeltal1(W,z,deltal2)
            derivW0 = np.dot(deltal1 , np.transpose(h[0]))

            W[2] -= learningRate * derivW2 #performs a backpropagation step using the calculated derivitives
            B[2] -= learningRate * deltal3
            W[1] -= learningRate * derivW1
            B[1] -= learningRate * deltal2
            W[0] -= learningRate * derivW0
            B[0] -= learningRate * deltal1

            print("Back-Propagation Done")
        except Exception as e:
            print(e)
            time.sleep(10)
        
    cont = input("Would you like to continue? (y/n) > ") #asks user if they want to continue
    if cont != 'y':
        break
    
print("Saving trained weights and biases . . .")
np.save("WeightsCNN", W) #saves the weights and biases
np.save("BiasCNN", B)    
        




