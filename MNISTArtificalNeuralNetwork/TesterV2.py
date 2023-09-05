"""
This prorgam allows the user to test the accuracy of their neural network after they have trained it. 
Use the files 'weights' and 'bias' for a reasonably accurate network
""" 


import numpy as np
import matplotlib.pyplot as plt
import random
import time

def f(x): #defines the sigmoid normalization function used in the program
    return 1 / (1 + np.exp(-x)) 

def readPred(predectedVal):  #reads the column vector the network outputs and determins what number it corresponds to 
    largestVal = 0
    index = 0
    for i in range (0,10):
        if (predectedVal[i] > largestVal):
            index = i
            largestVal = predectedVal[i]
    return index

def displayNumber(valueArray, index): #displays the MNIST dataset number as an image painted with text
    p = 0
    for i in range(0,27):
        print(" ")
        for j in range(0,28):
            temp = (x[index][p][0])
            if (temp == 0):
                print("  ", end=(''))
            else:
                print("XX",end=(''))
            p += 1

def displayPrediction(h):
    for t in range (0,3):
        print(h[3][t],end=(""))
    print("")
    for c in range(3,6):
        print(h[3][t],end=(""))
    print("")
    for c in range(6,9):
        print(h[3][t],end=(""))
    print("")
    print(h[3][9])
    
        

x = np.load("Xtest.npy") #Loads the MNIST test images files (thses are different to the dataset used to train the network)
y = np.load("Ytest.npy")

WeightFileName = str(input("What is the name of the Weight file you would like to use?> ")) + ".npy"
BiasFileName = str(input("What is the name of the Bias file you would like to use?> ")) + ".npy"


B = np.load(BiasFileName, allow_pickle = True) #loads user specified weights and bias files into the variables
W = np.load(WeightFileName, allow_pickle = True)

h = [0,0,0,0] #initializes array to hold the value of each network layer

setSize = int(input("How large would you like the sample to be? (max 10000)> "))
tmpOutputAns = input ("would you like the program to output all the test numbers in the sample? (y/n)> ")
if tmpOutputAns == "y":
    output = True
else:
    output = False

totalRight = 0 

setStart = random.randint(0,10000 - setSize) #starts sampling ast a random point in the set
for k in range(setStart, setStart + setSize): 

    h[0] = x[k]
    
    for i in range (0,3): #performs a forward propagation of the specific number in the test dataset
        h[i+1] = f( np.dot(W[i],h[i]) + B[i] )
       
    predectedVal = readPred(h[3])  #obtains the value of the prediction from the network
    actualVal = readPred(y[k])  #obtains the actual value from the dataset labels
 
    if output: 
        print("Calculating...")  #outputs information about the number and the accuracy of the networks guess
        displayNumber(x,k)
        print("\n \n ")
        
        #displayPrediction(h)
        
        print("Predected Value: " + str(predectedVal))
        print("Actual Value: " + str(actualVal))
        
        if(actualVal == predectedVal):
            totalRight += 1
            print("correct")
        else:
            print("incorrect")
            
        print("-" * 50)
      
    else:
        
        if(actualVal == predectedVal):
            totalRight += 1
            
        
        
            
print("\nThe network is: " + str(totalRight * 100 / setSize) + "% Accurate" ) #prints the network accuracy

