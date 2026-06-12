import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler
from markettesting.config import TICKER_DIR, BASE_DIRECTORY
from markettesting.formatting import pull_ticker_csv, pull_yf, check_invalid_ticker


def create_ticker_scaler(time_period, use_download=False, dump_location="StockModel.pkl"):
    scaler = RobustScaler()
    tickerList = pd.read_csv(TICKER_DIR)
    tickerList = tickerList['Symbol']

    comp_data_list = []

    for ticker in tickerList:
        print(f"CURRENT TICKER {ticker}")
        if use_download:
            ticker_data = pull_ticker_csv(ticker)
            if ticker_data is None:
                print(f"create_scaler : {ticker}.csv does not exist! Skipping")
                continue
        else:
            ticker_data = pull_yf(ticker, time_period=time_period)

        if check_invalid_ticker(ticker, ticker_data):
            print("Skipping...")
            continue

        comp_data_list.append(ticker_data)
        
    comp_data_list = pd.concat(comp_data_list, ignore_index=True)
        
    scaler.fit(comp_data_list)
    pickle.dump(scaler, open(BASE_DIRECTORY / dump_location, "wb"))

    return scaler
