# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:16:26 2023

@author: Matti

This program uses a tensorflow LSTM model to forecast the opening price of bitcoin an hour into the future. 
It first imports the data from a CSV. Then normalises the data and formats it into a training dataset.
The model is then trained and evaluated and a sample model prediciton is displayed.

The model takes an input of 100 timesteps with 5 features, these being opening price, closing price, hour high, hour low ans trading volume.
It outputs 1 data point.

To normalise the data I have calculated the 50 hour SMA and standard deviation. Then subtracted the SMA and divided by the standard deviation.
"""

#importing required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
import random

#this function formats the impoted data from the pandas dataframe.
def generateTrainingSets(inputSize, shift, labelSize, dataSet, numberOfSamples):
    dataSet = dataSet.to_numpy() #type casts dataframe to numpy array
    dataSet = dataSet.transpose()
    xTrain = [] #creates variables to hold he formatted data
    yTrain = []
    for i in range (0,numberOfSamples): #itterates through the amount of sampled required
        start = random.randint(0 , len(dataSet[0]) - inputSize - shift - labelSize) #selects a random starting position in the dataset
        tmpx = []
        for j in range (0, len(dataSet)): #itterates through the 5 features
            tmpx.append(dataSet[j][start : start + inputSize]) #extracts a time series of specified length to to append to the output dataset
            
        yTrain.append(dataSet[0][start + inputSize + shift : start + inputSize + shift + labelSize]) #extracts the relevent label data from the dataset
        xTrain.append(np.array(tmpx).transpose())  #transposes the array for the correct dimentionality
            
    return  xTrain, yTrain #returns formatted dataset

df = pd.read_csv('BTCUSDTDataset2.csv') #opens dataset
df = df.drop('Time', axis=1) #drops time enrty (I considered implementing a time signal column until I remembered BTC is traded online and the exchange never closes)


WindowSize = 50 #specify the length of time the SMA will be taken over

Volume = df['Volume'] #extract volume data from the set (this will be normalised differently)
maxVolume = df['Volume'].max() #finds max volume figure from the set

sma = df.rolling(window=WindowSize).mean() #clculates an array containing the SMA
std = df.rolling(window=WindowSize).std() #Calsulates the standard deviation in this set

dfScaled = (df - sma) / std #normalises price data
dfScaled['Volume'] = Volume / maxVolume #normalises volume daya
dfScaled.drop(index=df.index[:WindowSize], axis=0, inplace=True) #drops first data points, we have no SMA data here so cant normalize the first datapoints


inputLength = 100 #specifies input length for network
outputLength = 1 #specifies output length for the network
trainingDatasetBatchSize = 8000 #specifies size of sampled training data

xTrainingData,  yTrainingData = generateTrainingSets(inputLength, 1, outputLength, dfScaled,trainingDatasetBatchSize) #generates training dataset

xTrainingData = np.array(xTrainingData) #typecasts to numpy arrays
yTrainingData = np.array(yTrainingData)


"""
traindf = pd.DataFrame(xTrainingData[5], columns = ['Open','Close','High','Low','Volume']) #This plots the normalised training and label data
labeldf = pd.DataFrame(yTrainingData[5], columns = ['Open'])

plt.plot([i for i in range(0,inputLength)], traindf['Open'])
plt.plot([i for i in range(inputLength , inputLength + outputLength)], labeldf['Open'])
"""


model = Sequential()  #creates model

# Add LSTM layer
model.add(LSTM(64, activation='sigmoid', input_shape=(inputLength,5))) #ads LSTM layer with specified length and depth of 5 (for the 5 features of the data)

# Add output layer
model.add(Dense(64)) #ads a dense layer
model.add(Dense(outputLength)) #ads a dense output layer


# Compile the model
model.compile(optimizer='adam', loss='mse') 

model.summary() #outputs model summary

model.fit(xTrainingData, yTrainingData, epochs=8, batch_size = 1024, verbose=1) #trains model

score = model.evaluate(xTrainingData, yTrainingData, batch_size = 500) #evaluates models average loss over 500 samples (yes I know I should use different data from the training data here!)
print('Evaluation Score: ' + str(score)[:6]) #prints score

print("An example of a prediction will now display on another window") 

testInd = random.randint(0,trainingDatasetBatchSize) #generates a random test index
xTest = np.expand_dims(xTrainingData[testInd],0) #extracts a random sample from the dataset for use by the model
yTest = yTrainingData[testInd]  #extracts corresponding label
yPred = model.predict(xTest) #uses the model to make a prediciton
xTestPlot = xTest[0].transpose()[0] #formats time series for plotting

fig = plt.figure() #creates figure
ax = plt.subplot()
ax.plot([i for i in range(inputLength-20,inputLength)], xTestPlot[len(xTestPlot) - 20:], label='BTC opening price data') #plots price data
ax.plot([i for i in range(inputLength , inputLength + outputLength)], yTrainingData[testInd], 'bo', label = 'BTC opening price data point') #plots label
ax.plot([i for i in range(inputLength , inputLength + outputLength)], yPred[0], 'bx', label = 'Model Prediction') #plots model prediction
legend = ax.legend(loc='upper center', fontsize='large') #creates plot legend

plt.title("Mode Prediciton vs Actual Data") #creates plot title
plt.xlabel('Time (h)') #creates plot axis labels
plt.ylabel('Standard deviations away from SMA')

plt.show() #shows plot


     



