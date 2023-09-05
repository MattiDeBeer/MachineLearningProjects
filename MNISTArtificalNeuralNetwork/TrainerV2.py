""" 
This Program trains the network on the MNIST datasaet. It works by taking samples of the dataset and then 'guessing' the digit by performing a forward propagation. 
The cost (error) of each guess is then calculated and the nesesary derivitives w.r.t the weights and biases are then calculated and averaged over the entire sample. 
A backpropagation step is them performed using these averaged derivitives and the average error of the sample displayed. 
"""


#importing nescsarry libraries 
import numpy as np
import matplotlib.pyplot as plt
import random
import time

def f(x): #defining the sigmoid normalization function as f(x)
    return 1 / (1 + np.exp(-x))

def f_deriv(x): #defining the derivitive of the sigmoid function for us in the backpropagation algorithm
    return f(x)*(1-f(x))

def calcCostVector(predVal, actualVal): #defines a function to calculate the netowrks cost value in vector form using the predecided cost formula 
    costVector = 0.5*((predVal - actualVal) ** 2)
    return costVector

def calcDeltal3(predVal,z,actualVal): #defines a function to calculate delta2, the dervivtive of the bisa vectors on layer 3 
    return (predVal - actualVal) * f_deriv(z[2])

def calcDeltal2(weights,z,deltal3): #a finction to calculate delta2, the dervivtive of the bisa vectors on layer 2 
    return np.dot(np.transpose(weights[2]), deltal3) * f_deriv(z[1])

def calcDeltal1(weights,z,deltal2): #a finction to calculate delta1, the dervivtive of the bias vectors on layer 1
    return np.dot(np.transpose(weights[1]), deltal2) * f_deriv(z[0])

def sumVector(vector): #defines a function to sum all the values in a column vector
    total = 0
    for i in range (0, len(vector) -1):
        total += vector[i][0]
    return total

x = np.load("X.npy") #loads the x array (array of handwriten digits) into a variable 
y = np.load("Y.npy") #loads the y array (array of labels to the handwritten sigits) into a variable 

learningRate = float(input("What would you like the learning rate to be? (typically a value around 0.005 works well) > ")) #defines the networks learning rate 

W0 = np.array(np.random.uniform(-1, 1, (16,784))) #makes new randomized weights with approiate dimensions for the given layer
W1 = np.array(np.random.uniform(-1, 1, (16,16)))  #W0 is the weight matrix to calculate layer 1, W1 for layer 2 and so on  ...
W2 = np.array(np.random.uniform(-1, 1, (10,16)))

B0 = np.array(np.random.uniform(-1, 1,(16,1))) #makes the bias vectors with appropiate dimensions for the given layer
B1 = np.array(np.random.uniform(-1, 1,(16,1))) #B0 is the bias vector to calculate layer 1, B1 to calculate layer 2 ...
B2 = np.array(np.random.uniform(-1, 1,(10,1)))

h = [0,0,0,0] #defines h, the variable that holds the value of the neurons (column vector) of each layer
z = [0,0,0] # defines z, the variable that the value of the neurons before their put through the sigmoid function (used for backpropagations)


setSize = int(input("How large would you like your samples to be (max 49999) > ")) #defines the amount of images the derivitives are averaged over before a back propagation change is made to the weights and biases

W = [W0,W1,W2] #defines W, an array of all the weight vectors
B = [B0,B1,B2] #defines B, an array of all the bias vectors

xAxis = [] #defines the y axis of the output (the cost of the network)
yAxis = [] #defines the x axis (itteration number)

itterationNum = int(input("How many backpropagation steps would you like to perform? > "))

for p in range (0,itterationNum): #p defines the amount of itterations the program will perform until its done training

    print("Itteration: " + str(p)) #prints itteration number
    costSum = 0 
    
    derivW2Total = 0 #initialises runing total variables for the derivitives (Yes i know its clunky, I was 16 when I wrot this!)
    deltal3Total = 0
    derivW1Total = 0
    deltal2Total = 0
    derivW0Total = 0
    deltal1Total = 0
        
    sampleStart = random.randint(0,50000 - setSize) #chooses a random place in the sample  array to start a sample
    for j in range(sampleStart , sampleStart + setSize): #Itterates through a pre specified number of handwritten digits from the randomly chosen start
        
        h[0] = x[j] #sets the network input variable
    
        for i in range (0,3): #preforms a forward proagation through the network to obtain a prediction of the handwritten digit
            z[i] = np.dot(W[i],h[i]) + B[i]
            h[i+1] = f(z[i])
            
        costVector = calcCostVector(h[3], y[j]) #Calculates the cost (error) of the network from the label of the digit
        costSum += sumVector(costVector)
        
        #Below is the backpropagation algorithm
        
        deltal3 = calcDeltal3(h[3],z,y[j]) #Here the derivitives w.r.t the weights and biases of each layer is calculated
        derivW2 = np.dot(deltal3 , np.transpose(h[2])) #derivw2 is the derivitive of layer 3s weight matrix
            
        deltal2 = calcDeltal2(W,z,deltal3) #deltal2 is the derivitive of layer 3s bias vector
        derivW1 = np.dot(deltal2 , np.transpose(h[1])) 
            
            
        deltal1 = calcDeltal1(W,z,deltal2)
        derivW0 = np.dot(deltal1 , np.transpose(h[0]))
        
        derivW2Total += derivW2  #This block of code appends the calculated derivitives to relevent variables (this is in order to calculate the average gradient descent step to perform over the whole sample)
        deltal3Total += deltal3
        derivW1Total += derivW1
        deltal2Total += deltal2
        derivW0Total += derivW0
        deltal1Total += deltal1
    
    W[2] -= learningRate * 1/ setSize * derivW2Total #Once the total gradient descent 'direction' is found over the whole sample the weights and biases are updated here.
    B[2] -= learningRate * 1/ setSize * deltal3Total
    W[1] -= learningRate * 1/ setSize * derivW1Total
    B[1] -= learningRate * 1/ setSize * deltal2Total
    W[0] -= learningRate * 1/ setSize * derivW0Total
    B[0] -= learningRate * 1/ setSize * deltal1Total
    
    xAxis.append(p) 
    yAxis.append(costSum/setSize) #Prints average cost (error) of the run sample. This will decrease over time once the netwoke becomes more accurate 
    print(costSum/setSize)

plt.plot(xAxis,yAxis) #Plots the network cost af a function of backrpopagaion step number
plt.show()

weightsFilename = input("what would you like to save your weights file as? > ")   
biasFilename = input("what would you like to save your bias file as? > ")     

np.save(weightsFilename, W) #saves weights and biases in a numpy variable for later use
np.save(biasFilename, B)
    

   
    
    

