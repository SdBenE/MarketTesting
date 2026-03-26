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
