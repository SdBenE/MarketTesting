"""
Global Configuration of file paths
between methods
"""

from pathlib import Path

BASE_DIRECTORY = Path(__file__).parent
DATA_FOLDER_DIR = BASE_DIRECTORY / "data_folder"
TICKER_DIR = BASE_DIRECTORY / "tickers.csv"
EXTERNAL_FACTORS_DIR = DATA_FOLDER_DIR / "external_factors"