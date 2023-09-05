import imageReader
import numberPainter
import time
import numpy as np

"""This program uses the previously trained weights and biases to guess a users handwritten digit. If the
 guess is incorrect the program will perform a backpropagation step to update the networks weights and biases."""

def f(x):
    return 1 / (1 + np.exp(-x)) #Defines sigmoid normalization function

def f_deriv(x):
    return f(x)*(1-f(x)) #Defines the derivitive of the sigmoid normalization function (used in the backpropagation steps)

def calcCostVector(predVal, actualVal): #Defines a function to calculate the cost (error) of each prediction
    costVector = 0.5*((predVal - actualVal) ** 2)
    return costVector

def calcDeltal3(predVal,z,actualVal): #Calculates the derivitive of the third layer of the network (for backpropagation)
    return (predVal - actualVal) * f_deriv(z[2]) #This function and the below functions use the chain rule to progressively calculate the required derivitives

def calcDeltal2(weights,z,deltal3): #Calculates the derivitive of the second layer of the network (for backpropagation)
    return np.dot(np.transpose(weights[2]), deltal3) * f_deriv(z[1])

def calcDeltal1(weights,z,deltal2): #Calculates the derivitive of the first layer of the network (for backpropagation)
    return np.dot(np.transpose(weights[1]), deltal2) * f_deriv(z[0])

def sumVector(vector): #Sums all values in a vector (used to calculate the total error)
    total = 0
    for i in range (0, len(vector) -1):
        total += vector[i][0]
    return total
    
def backPropagate(y,W,B,z,h,learningRate):  #Defines the backpropagation algorithm

    
    deltal3 = calcDeltal3(h[3],z,y)  
    derivW2 = np.dot(deltal3 , np.transpose(h[2]))
            
    deltal2 = calcDeltal2(W,z,deltal3)
    derivW1 = np.dot(deltal2 , np.transpose(h[1]))
            
    deltal1 = calcDeltal1(W,z,deltal2)
    derivW0 = np.dot(deltal1 , np.transpose(h[0]))

    W[2] -= learningRate * derivW2 #Weights and biases are uppdated in this block with the derivitives calculated above
    B[2] -= learningRate * deltal3
    W[1] -= learningRate * derivW1
    B[1] -= learningRate * deltal2
    W[0] -= learningRate * derivW0
    B[0] -= learningRate * deltal1    

    return (W,B)

def generateNumberVector(value): 
    vector = np.zeros((10,1))
    vector[value][0] = 1.0
    return vector
    
    

def forwardPropagate(x,W,B): #This function performs a forward propagation steps, takes arguments (input image vector, array of weight matricies, array of bias matricies)
    h = [0,0,0,0] #An arrya of the value of the neurons between layers
    z = [0,0,0] #An array of the values of the nuurons between each layer (before the normalization is applied)

    h[0] = x #sets the network input vector (layer 0)
    
    for i in range (0,3): #forward propagates the above input through the network
        z[i] = np.dot(W[i],h[i]) + B[i] 
        h[i+1] = f(z[i])

    return(h[3],W,B,z,h) #returns the network output, weights and biases used and neuron value for each layer (required for backpropagation)

def readPred(predectedVal): #defines a function to calculate what the predicted number is from the network output layer 
    largestVal = 0
    index = 0
    for i in range (0,10):
        if (predectedVal[i] > largestVal):
            index = i
            largestVal = predectedVal[i]
    return index

while True:
    try:
        numberPainter.paintNumber() #Opens number painter window, saves painted image locally
        Xraw = imageReader.convertFile() #converts painted image file into a column vector that can be inputted into the matrix

        B = np.load("Bias1.npy", allow_pickle = True) #loads pretrained weight and bias files 
        W = np.load("Weights1.npy", allow_pickle = True)

        netOut = forwardPropagate(Xraw,W,B) #performs a forward propagation to predict what the number is
        predictedNumber = readPred(netOut[0]) 
        print ("Network Predicted Value: " + str(predictedNumber)) #outputs prediction


        correct = input("correct? (y/n)> ")
        if (correct == "n"): #If incorrect the program will ask for the the correct answer and perform a backpropagation step with the correct answer
            actualValue = int(input("input correct value > "))
            actualNumberVector = generateNumberVector(actualValue)
            costVector = calcCostVector(netOut[0], actualNumberVector)
            cost = sumVector(costVector)
            print("Cost: " + str(cost))
            
            print("")
            for i in range (0,10):
                l = len(str(netOut[0][i]))
                j = 20 - l
                print(str(i) + ": " + str(netOut[0][i]) + " " * j + str(actualNumberVector[i]))
            print("")
            backOut = backPropagate(actualNumberVector,netOut[1],netOut[2],netOut[3],netOut[4],0.005)
            W = backOut[0]
            B = backOut[1]
            print("BackPropagation Done! ")
            
        elif (correct == "y"): #If correct the program will preform a backpropagation step with the predicted answer
            actualNumberVector = generateNumberVector(predictedNumber)
            backOut = backPropagate(actualNumberVector,netOut[1],netOut[2],netOut[3],netOut[4],0.1)
            W = backOut[0]
            B = backOut[1]
            print("BackPropagation Done! ")

        else:
            print("Invalid Option")
            print("saving...")
            time.sleep(1)
            np.save("Bias1.npy",B) #The updated weights and biaes will then be saved
            np.save("Weight1.npy",W)
            break;
            
    except Exception as e:
        print(e)
        time.sleep(100)
        
    close = input("Would you like to continue? (y/n) > ")
    if close != "y":
        print("saving...")
        np.save("Bias1.npy",B) #The updated weights and biaes will then be saved
        np.save("Weight1.npy",W)
        break
            
        
        
        

