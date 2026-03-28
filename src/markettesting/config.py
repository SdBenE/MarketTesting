"""
Global Configuration of file paths
between methods
"""

from pathlib import Path

BASE_DIRECTORY = Path(__file__).parent
DATA_FOLDER_DIR = BASE_DIRECTORY / "dataFolder"
TICKER_DIR = BASE_DIRECTORY / "tickers.csv"
