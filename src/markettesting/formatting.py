"""
Training data formatting, graphing, and downloading
"""

import time
import os
import yfinance as yf
import pandas as pd
from markettesting.config import DATA_FOLDER_DIR, TICKER_DIR

def file_formation(period_years=1, download=False):
    """
    Retrive training files for use in model training
    """
    data_list = pd.read_csv(TICKER_DIR)
    temp_data_list = data_list
    print(data_list)
    symbols = data_list['Symbol']


    duration = f"{period_years}y"

    # emptyTickers = []

    # if not os.path.exists("/MarketTesting/src/markettesting/dataFolder"):
    #     os.makedirs("/MarketTesting/src/markettesting/dataFolder", exist_ok=True)

    for ticker in symbols:
        if os.path.exists(DATA_FOLDER_DIR / f'{ticker}.csv'):
            print(f'File {ticker}.csv already exists! Skipping...')
            continue

        print(f"      CURRENT TICKER {ticker}      ")

        start = time.perf_counter()

        my_ticker = yf.download(ticker, period=duration)
        # my_ticker.columns = [None] * len(my_ticker.columns)

        print(f"Download completed in {time.perf_counter() - start} [s]")

        if my_ticker.empty or len(my_ticker) < 10:
            print(f"[[{ticker}]] : TOOOOOO SMALLLLLLLL BYE BYE")
            # emptyTickers.append(ticker)

            drop_index = temp_data_list[data_list['Symbol'] == ticker].index
            temp_data_list = temp_data_list.drop(drop_index)
        elif download:
            my_ticker.to_csv(DATA_FOLDER_DIR / f'{ticker}.csv')

        # print(emptyTickers)
        print(f"data_list Length: {len(temp_data_list)}")
        # print(temp_data_list)

    temp_data_list.to_csv(TICKER_DIR)

def flatten_from_yf(data_set):
    """Flattens data to a format acceptable for StockModel-based models"""
    data_set = data_set.reset_index(drop=True)
    data_set.columns = data_set.columns.get_level_values(0)
    data_set.columns = ['Close', 'High', 'Low', 'Open', 'Volume']

    return data_set