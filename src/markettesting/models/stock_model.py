"""
STOCK MODEL CLASS
"""
import os
import yfinance as yf
import numpy as np
import pandas as pd
import pickle
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from markettesting.config import BASE_DIRECTORY, DATA_FOLDER_DIR, TICKER_DIR
from markettesting.formatting import pull_csv, pull_yf

class StockModel:
    """
    DEFAULT STOCK MODEL FOR MARKETTESTING
    """
    def __init__(self, sequence_length=100, num_features=5, time_period ='4y', model_import=False):
        """
        Default constructor for StockModel
        """
        self.time_period = time_period
        self.model = Sequential()
        self.epochs = 25
        self.units = 1000
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.scaler = None

        if model_import:
            self.import_model()
            self.import_pickle_scaler()
            print("Importing model from StockModel.keras")
        else:
            self.create_model()

        self.early_stop_system = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

    def import_model(self, model_directory="StockModel.keras"):
        """Imports model in .keras format"""
        self.model = load_model(BASE_DIRECTORY / model_directory)

    def import_pickle_scaler(self, pickle_directory="StockModel.pkl"):
        self.scaler = pickle.load(open(BASE_DIRECTORY / pickle_directory, "rb"))

    def create_model(self):
        """
        Creates default LSTM model for StockModel Class
        Stored in self.model
        """
        #Layer 1
        self.model.add(LSTM(units=self.units,
                            return_sequences=True,
                            input_shape=(self.sequence_length, self.num_features)))
        self.model.add(Dropout(0.2))

        #Layer 2
        self.model.add(LSTM(units=self.units, return_sequences=True))
        self.model.add(Dropout(0.2))

        #Layer 3
        self.model.add(LSTM(units=self.units, return_sequences=False))

        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def create_scaler(self, use_download=False, dump_location="StockModel.pkl"):
        self.scaler = StandardScaler()
        tickerList = pd.read_csv(TICKER_DIR)
        tickerList = tickerList['Symbol']

        comp_data_list = []

        for ticker in tickerList:
            print(f"CURRENT TICKER {ticker}")
            if use_download:
                ticker_data = pull_csv(ticker)
                if ticker_data is None:
                    print(f"create_scaler : {ticker}.csv does not exist! Skipping")
                    continue
            else:
                ticker_data = pull_yf(ticker, time_period=self.time_period)

            if self.check_invalid_ticker(ticker, ticker_data):
                print("Skipping...")
                continue

            comp_data_list.append(ticker_data)
        
        comp_data_list = pd.concat(comp_data_list, ignore_index=True)
        
        #TODO: Remove before use
        print(comp_data_list.dtypes)
        print(comp_data_list.head())
        print(comp_data_list.describe())
        
        self.scaler.fit(comp_data_list)
        pickle.dump(self.scaler, open(BASE_DIRECTORY / dump_location, "wb"))

    def get_prediction(self, input_data):
        """Getter for predictions"""
        # if input_data.shape() != (self.sequence_length,1,self.num_features): #TODO: Add thrower for invalid shape
        #     raise ValueError("get_prediction: Incorrect sequence_length")

        #SCALE AND RESHAPE TO TENSOR
        scaled_input = self.scaler.transform(input_data)
        scaled_input = np.expand_dims(scaled_input, 0)
        
        #PREDICT
        scaled_prediction = self.model.predict(scaled_input)
        print("Scaled prediction", scaled_prediction)

        print("Scaler mean", self.scaler.mean_)
        print("Scaler scale", self.scaler.scale_)

        #FIT WITH DUMMY, DESCALE
        placeholder = np.zeros((1,5))
        placeholder[0,0] = scaled_prediction
        print(placeholder)
        return self.scaler.inverse_transform(placeholder)[0,0]
    
    def data_sequence(self, training_data):
        """
        Sequences time-series data windows for training
        Bases the size of the windows on stored sequence length field
        """
        x_list = []
        y_list = []
        i = 0

        for j in range(self.sequence_length, len(training_data)):
            x_list.append(training_data.iloc[i:j])

            y_list.append(training_data.at[j, 'Close']) #Adds the following day to it
            i+=1

        return np.array(x_list), np.array(y_list)

    def check_invalid_ticker(self, ticker, raw_data):
        if raw_data is None or len(raw_data) < 100:
            print(f"Ticker {ticker} is too small or doesn't exist")
            return True
        if raw_data['Close'].median() > 500:
            print(f"Closing prices for {ticker} are extremely high, likely index")
            return True
        if raw_data['Close'].std() > 500:
            print(f"Price variance for {ticker} are extremely high")
            return True
        else:
            return False

    def train_model(self, use_download=True, save_dir="StockModel", batch_size=128):
        """
        StockModel preprocessing and data push-through
        """
        if self.scaler == None:
            if use_download:
                self.create_scaler(use_download=use_download)
            else:
                self.import_pickle_scaler()

        data_list = pd.read_csv(TICKER_DIR)
        data_list = data_list['Symbol']

        for ticker in data_list:
            print(f'     Current Ticker: {ticker}')

            if use_download:
                raw_data = pull_csv(ticker)
            else:
                raw_data = pull_yf(ticker, time_period=self.time_period)
            
            if raw_data is None:
                print(f"{ticker} not found!")
                print("Skipping...")
                continue

            raw_data = raw_data.apply(pd.to_numeric, errors='coerce').dropna()

            if self.check_invalid_ticker(ticker, raw_data):
                print("Skipping...")
                continue

            #SCALING
            scaled_data = self.scaler.transform(raw_data)
            scaled_data = pd.DataFrame(scaled_data, columns=raw_data.columns)

            #WINDOWING
            x_full, y_full = self.data_sequence(scaled_data)

            #SPLITTING DATA
            split_index = int(0.8 * len(y_full)) #Integer casting for proper index

            x_train = x_full[:split_index]
            y_train = y_full[:split_index]
            x_test = x_full[split_index+1:]
            y_test = y_full[split_index+1:]

            self.model.fit(
                x_train,
                y_train,
                epochs=self.epochs,
                batch_size=batch_size,
                callbacks=[self.early_stop_system],
                validation_data=(x_test, y_test)
            )

            self.model.save(BASE_DIRECTORY / f'{save_dir}.keras')
