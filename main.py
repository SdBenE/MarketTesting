import yfinance as yf
import csv
from pathlib import Path

myTestTicker = yf.Ticker("AAPL")

print(myTestTicker.info)
print(myTestTicker.history(period='max'))

myData = yf.download('AAPL', period='1mo')
myPath = Path("myData.txt")

Path.write_text("Hello")