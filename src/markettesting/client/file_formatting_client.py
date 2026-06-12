"""
Base client code for file downloading
"""

import warnings
from markettesting.formatting import file_formation, volatility_file_formation

warnings.filterwarnings('ignore')

volatility_file_formation(period_years='max', download=True)
file_formation(period_years=10, download=True)