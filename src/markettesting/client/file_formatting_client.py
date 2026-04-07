"""
Base client code for file downloading
"""

import warnings
from markettesting.formatting import file_formation

warnings.filterwarnings('ignore')

file_formation(period_years='max', download=True)