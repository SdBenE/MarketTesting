import yfinance as yf
import csv
import time
import sys
#from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import dataFinder
import tensorflow as tf
import formatting
import keras
from datetime import date
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

class LTSMModel:
    def __init__(self, epochs=25, units=50):
        self.model = Sequential()
        self.epochs = epochs
        self.units = units
        self.predictionTable = None
    
    def predictions(self, date=date.today()):
        # if self.tickerTable != None: #TODO: Uncomment this section once else statement is completed
            
        # else:
            tickerList = pd.read_csv('/home/enjamin_lmore/tf-env/MarketTesting/src/markettesting/tickers.csv')
            tickerList = tickerList['Symbol']
            
            self.predictionTable = pd.DataFrame()
            self.predictionTable['Ticker'] = tickerList

            print(self.predictionTable.head())


            self.predictionTable[f'{date.month}-{date.day}-{date.year}'] = None
            print(self.predictionTable.head())
            # print(today)



    def importModel(self, filename='itWorked.keras'):
        self.model = load_model(filename)

    def createModel(self, durationYears=1, sequenceLength=100):
        self.timePeriod = f"{durationYears}y"
        self.sequenceLength = sequenceLength

        #Layer 1
        self.model.add(LSTM(units=self.units, return_sequences=True, input_shape=(self.sequenceLength, 1)))
        self.model.add(Dropout(0.2))

        #Layer 2
        self.model.add(LSTM(units=self.units, return_sequences=True))
        self.model.add(Dropout(0.2))
        
        # self.model.add(LSTM(units=self.units, return_sequences=True))
        # self.model.add(Dropout(0.2))

        # #Layer 3
        # self.model.add(LSTM(units=self.units, return_sequences=True))
        # self.model.add(Dropout(0.2))

        # #Layer 4
        # self.model.add(LSTM(units=self.units, return_sequences=True))
        # self.model.add(Dropout(0.2))

        #Layer 5
        self.model.add(LSTM(units=self.units, return_sequences=False))

        self.model.add(Dense(units=1))

        self.earlyStopSystem = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def dataSequence(self, trainingData):
        xList = []
        yList = []
        
        i = 0

        for j in range(self.sequenceLength, len(trainingData)):
            xList.append(trainingData.iloc[i:j]) #Adds dataframe between start at i and i the sequence length, all the way to the end of the training data
            # print(f'Length of xList at first index: {len(xList[0])}')

            yList.append(trainingData.at[j, 'Close']) #Adds the following day to it
            # print(f'j : {j}')
            # print(f'i : {i}')
            i+=1
            # break #TODO: Remove this!!!!!

        return np.array(xList), np.array(yList)

    def trainModel(self, useDownload=True):
        dataList = pd.read_csv('/home/enjamin_lmore/tf-env/MarketTesting/src/markettesting/tickers.csv')
        dataList = dataList['Symbol']

        for ticker in dataList:
            print(f'     Current Ticker: {ticker}')

            if useDownload:
                if os.path.exists(f'dataFolder/{ticker}.csv'):
                    rawData = pd.read_csv(f'dataFolder/{ticker}.csv')
                else:
                    print(f'Ticker file {ticker}.csv is missing! Skipping...')
                    continue
            else:
                rawData = yf.download(ticker, period=self.timePeriod)

            # rawData = yf.download(ticker, period=self.timePeriod)

            MIN_DATA_POINTS = self.sequenceLength + 10

            if len(rawData['Close']) < MIN_DATA_POINTS or rawData.empty:
                print("CATCH 0: SKipping...")
                continue

            # print(rawData.columns)
            # print(rawData.head())
            for x in range(0,2):
                rawData = rawData.drop(index=x)
            # print(rawData.head())

            rawData.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

            if rawData.empty:
                print(f"Catch 1: Ticker {ticker} is empty Skipping...")
                continue

            rawData.dropna(subset=['Close'], inplace=True)
            rawData = rawData.reset_index()
            rawData.drop(labels='index', axis=1, inplace=True)
            # print(rawData.head())

            rawData.drop(labels='Date', axis=1, inplace=True)
            # print(rawData.head()) 

            #SCALING

            scaler = MinMaxScaler(feature_range=(0,1))

            scaledData = scaler.fit_transform(rawData)
            scaledData = pd.DataFrame(scaledData, columns=rawData.columns)

            # print(scaledData.head())

            #ORGANIZING

            xFull, yFull = self.dataSequence(scaledData)
            # print(xTraining)
            # print(yTraining)
            # print(xTraining.shape)
            # print(yTraining.shape)

            #SPLITTING DATA
            splitIndex = int(0.8 * len(yFull)) #Integer casting for proper index

            xTrain = xFull[:splitIndex]
            yTrain = yFull[:splitIndex]
            xTest = xFull[splitIndex+1:]
            yTest = yFull[splitIndex+1:]

            history = self.model.fit(
                xTrain, 
                yTrain, 
                epochs=self.epochs, 
                batch_size=32, 
                callbacks=[self.earlyStopSystem],
                validation_data=(xTest, yTest)
            )

            print(f'CLASS SIZE: {sys.getsizeof(self)} bytes')
            print(f'MODEL SIZE: {sys.getsizeof(self.model)} bytes')

            self.model.save('itWorked.keras')
