# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:16:23 2020

@author: Matti

This Program gets prices and marked volume using the Binance API and saves it as a csv file
"""
from binance.client import Client
import time
import numpy as np

def printServerTime(): #a function to print the server time
    time_res = client.get_server_time()  #gets time from server
    time_stamp = str(time_res['serverTime']) 
    time_stamp_fore = time_stamp[:10] #shortens time to 10 digits
    date_time = time.ctime(int(time_stamp_fore)) #converts to date and time format
    print(date_time) #prints time
     
def formatKlines(klines): #this program formats the data obtained form Binance
    cTime = []
    cOpen = [] #creates variables to store data in
    cClose = []
    cHigh = []
    cLow = []
    cVolume = []

    for stick in klines: #appends data from server to arrays
        cTime.append(float(stick[0]))
        cOpen.append(float(stick[1]))
        cClose.append(float(stick[2]))
        cHigh.append(float(stick[3]))
        cLow.append(float(stick[4]))
        cVolume.append(float(stick[5]))
    
    return [cOpen, cClose, cHigh, cLow, cVolume, cTime] #returns 2d array of data
    

api_key = "put your key here" #sets api keys (I havent included my actual keys for obvious reasons)
api_secret = "put you key here"

client = Client(api_key, api_secret)  #opens a client with keys 

printServerTime() #prints server time

klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "1 year ago UTC") #gets price data from server, hourly prices for the last year
data_set = formatKlines(klines)  #calls data formatting function
data_set = np.transpose(data_set) #transposes data for storage in csv file
np.savetxt('BTCUSDTDataset2.csv', data_set, delimiter=',', header='Open,CLose,High,Low,Volume,Time', comments='')  #saves data to csv file and adds headers
print("BTC collected")  #print that the operation is completed






