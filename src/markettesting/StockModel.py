"""
STOCK MODEL CLASS
"""
import os
import yfinance as yf
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from config import BASE_DIRECTORY, DATA_FOLDER_DIR, TICKER_DIR

class StockModel:
    """
    DEFAULT STOCK MODEL FOR MARKETTESTING
    """
    def __init__(self, sequence_length=100, num_features=5, time_period ='4y'):
        """
        Default constructor for StockModel
        """
        self.time_period = time_period
        self.model = Sequential()
        self.epochs = 25
        self.units = 1000
        self.num_features = num_features
        self.sequence_length = sequence_length

        self.create_model()

        self.early_stop_system = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

    def import_model(self):
        """Imports model in .keras format"""
        self.model = load_model(BASE_DIRECTORY / f"StockModel.keras")

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

    def get_prediction(self, input_data):
        """Getter for predictions"""
        if len(input_data) != self.sequence_length:
            raise ValueError("get_prediction: Incorrect sequence_length")
    
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

    def pull_csv(self, ticker, main_dir=DATA_FOLDER_DIR):
        """Pulls downloaded stock data in .csv format"""
        csv_dir = main_dir / f"{ticker}.csv"
        if os.path.exists(csv_dir):
            data_set = pd.read_csv(csv_dir)
        else:
            return None

        data_set = data_set.iloc[2:].copy()
        data_set.columns = data_set.columns.get_level_values(0)
        data_set = data_set.reset_index(drop=True)

        data_set = data_set.drop(labels='Price', axis=1)
        data_set.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        return data_set
    
    def pull_yf(self, ticker):
        """
        Fetches stock data from yfinance api
        Reformats table to match desired form for
        train_model
        """
        data_set = yf.download(ticker, period=self.time_period)
        data_set = data_set.reset_index(drop=True)

        if len(data_set) < 10:
            return None

        data_set.columns = data_set.columns.get_level_values(0)
        data_set.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        return data_set

    def train_model(self, use_download=True, save_dir="StockModel", batch_size=128):
        """
        StockModel preprocessing and data push-through
        """

        data_list = pd.read_csv(TICKER_DIR)
        data_list = data_list['Symbol']

        for ticker in data_list:
            print(f'     Current Ticker: {ticker}')

            if use_download:
                raw_data = self.pull_csv(ticker)
            else:
                raw_data = self.pull_yf(ticker)

            if raw_data is None or len(raw_data) < 100:
                print(f"Ticker {ticker} is too small or doesn't exist")
                print(f"Skipping {ticker}...")
                continue

            #SCALING
            # scaler = MinMaxScaler(feature_range=(0,1))
            scaler = StandardScaler()

            #This uses global scaling based on the whole data_set to check proper values

            # scaled_data = self.valueScaler.transform(raw_data)
            scaled_data = scaler.fit_transform(raw_data)
            scaled_data = pd.DataFrame(scaled_data, columns=raw_data.columns)

            #ORGANIZING
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