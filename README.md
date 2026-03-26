# MarketTesting

A Python project that uses **TensorFlow** and **yfinance** to fetch historical stock market data and build predictive models for price forecasting.

---

## Overview

MarketTesting is a learning-oriented project aimed at exploring machine learning techniques for stock market analysis. It pulls real financial data via yfinance and feeds it into TensorFlow-based models to generate predictions.

---

## Project Structure

```
MarketTesting/
├── src/
│   └── markettesting/      # Core package
│       └── ...             # Source modules
├── .gitignore
├── pyproject.toml          # Project metadata and dependencies
└── README.md
```

---

## Dependencies

- [TensorFlow](https://www.tensorflow.org/) — machine learning framework used for building and training prediction models
- [yfinance](https://github.com/ranaroussi/yfinance) — fetches historical and real-time stock market data from Yahoo Finance

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/SdBenE/MarketTesting.git
cd MarketTesting
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install the package

```bash
pip install -e .
```

---

## Learning Resources

### TensorFlow
- [Official TensorFlow Documentation](https://www.tensorflow.org/learn)
- [TensorFlow Tutorials — Time Series Forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Keras Sequential Model Guide](https://keras.io/guides/sequential_model/)
- [Deep Learning with Python (book by François Chollet)](https://www.manning.com/books/deep-learning-with-python)

### yfinance
- [yfinance GitHub & Docs](https://github.com/ranaroussi/yfinance)
- [yfinance PyPI Page](https://pypi.org/project/yfinance/)
- [Fetching Stock Data with yfinance (tutorial)](https://algotrading101.com/learn/yfinance-guide/)

### General Financial ML
- [Towards Data Science — Stock Prediction with LSTM](https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model)
- [Machine Learning for Trading (book by Stefan Jansen)](https://www.oreilly.com/library/view/machine-learning-for/9781839217715/)

---

## Roadmap

- [ ] Fetch historical stock data using yfinance
- [ ] Preprocess and normalize data for model input
- [ ] Build a baseline TensorFlow model (e.g., LSTM or Dense)
- [ ] Evaluate model performance on test data
- [ ] Visualize predictions vs. actual prices
- [ ] Experiment with additional features (volume, indicators, etc.)

---

## Disclaimer

This project is for **educational purposes only**. Nothing in this repository constitutes financial advice. Do not use model outputs to make real investment decisions.

---
