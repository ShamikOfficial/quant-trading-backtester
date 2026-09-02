# Quant Trading Backtester

[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python&logoColor=white)]()

> End-to-end algorithmic trading pipeline — data collection, XGBoost signals, backtesting, and performance analytics.

---

## Problem

Building a quantitative trading strategy requires more than a model — you need reliable data ingestion, feature engineering, signal generation, simulated execution, and rigorous performance evaluation in one reproducible pipeline.

## Solution

A modular Python framework implementing the **Signal → Order → Execute** flow with technical indicators, XGBoost-based ML strategies, and comprehensive backtesting metrics (Sharpe, Sortino, drawdown).

## Key Results

| Component | Capability |
|-----------|------------|
| Data collection | API ingestion with checkpoint/resume |
| Strategies | Technical indicators + XGBoost ML models |
| Backtesting | Sharpe, Sortino, drawdown analysis |
| Architecture | Modular `src/` pipeline |

---

## Features

- **Data Collection**: Automated data collection from APIs with checkpoint/resume capability
- **Data Processing**: Cleaning, normalization, and feature engineering
- **Trading Strategies**: Technical indicators and ML-based strategies (XGBoost)
- **Backtesting**: Comprehensive backtesting framework with performance metrics
- **Performance Analytics**: Sharpe ratio, Sortino ratio, drawdown analysis, and more

## Architecture

The system follows a modular architecture with the **Signal → Order → Execute** flow:

```
Data Layer → Signal Engine → Order Management → Execution → Analytics
```

### Components

- **Data Layer** (`src/data_loader.py`): Data loading, cleaning, and preprocessing
- **Signal Engine** (`src/strategies/`): Strategy implementations (technical indicators, ML models)
- **Mock Environment** (`src/simulator.py`): Portfolio and trade execution simulation
- **Analytics** (`src/evaluator.py`): Performance metrics and evaluation

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd LAB4-Algorithmic-Trading
```

2. Create and activate virtual environment:

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start

### 1. Stock Selection

Select stocks from S&P 500 with equal sector representation:

```bash
python src/stock_selector.py --n-stocks 20 --output-dir data/selections
```

### 2. Data Collection

Collect raw market data (requires API credentials):

```bash
# Set API credentials
export API_BASE=https://api.polygon.io
export API_KEY=YOUR_API_KEY

# Collect raw data
python run_data_collection.py --collect-raw \
  --selection-file data/selections/selected_stocks_20_*.json \
  --n-weekdays 60 --bar-minutes 1
```

### 3. Data Processing

Process raw data to add technical indicators:

```bash
python run_data_collection.py --process
```

### 4. Model Training

Train XGBoost models on processed data:

```bash
python run_ml_training.py --processed-file "data/processed/processed_*.csv" --mode train-all
```

### 5. Backtesting

Run backtests using trained models:

```bash
python run_backtest.py --processed-file "data/processed/processed_*.csv" \
  --ticker AVGO --initial-cash 1000000 --plot
```

## Usage

### Stock Selection

Select stocks with equal sector representation from S&P 500:

```bash
python src/stock_selector.py --n-stocks 20 --output-dir data/selections
```

The selection is saved as a JSON file in the specified directory.

### Data Collection

The data collection system supports checkpoint/resume functionality:

```bash
# Collect raw data with checkpoint support
python run_data_collection.py --collect-raw \
  --selection-file data/selections/selected_stocks_20_*.json \
  --n-weekdays 60 --bar-minutes 1 \
  --api-base https://api.polygon.io --api-key YOUR_API_KEY

# Process raw data
python run_data_collection.py --process

# Check status
python run_data_collection.py --status

# Clear checkpoints
python run_data_collection.py --clear-checkpoints
```

**API Credentials**: Set via environment variables or command-line arguments:
```bash
export API_BASE=https://api.polygon.io
export API_KEY=YOUR_API_KEY
```

### Model Training

Train XGBoost models on processed data:

```bash
# Train models for all tickers
python run_ml_training.py --processed-file "data/processed/processed_*.csv" --mode train-all

# Train model for specific ticker
python run_ml_training.py --processed-file "data/processed/processed_*.csv" \
  --mode train --ticker AVGO

# Generate predictions
python run_ml_training.py --processed-file "data/processed/processed_*.csv" \
  --mode predict --ticker AVGO
```

### Backtesting

Run backtests with various configurations:

```bash
# Single ticker backtest
python run_backtest.py --processed-file "data/processed/processed_*.csv" \
  --ticker AVGO --initial-cash 1000000

# Multiple tickers
python run_backtest.py --processed-file "data/processed/processed_*.csv" \
  --tickers AVGO HRL GL --initial-cash 1000000

# With date range and custom parameters
python run_backtest.py --processed-file "data/processed/processed_*.csv" \
  --ticker AVGO --start-date 2024-10-01 --end-date 2024-12-31 \
  --min-confidence 0.7 --lookback-window 20 --plot
```

Results are saved to `backtest_results/` directory in JSON format.

## Programmatic Usage

### Data Collection

```python
from run_data_collection import DataCollectionRunner

runner = DataCollectionRunner(
    api_base="https://api.polygon.io",
    api_key="your_api_key"
)

# Collect raw data
raw_data = runner.collect_raw_data(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    n_weekdays=60,
    bar_minutes=1,
    resume=True
)

# Process data
processed_data = runner.process_data(resume=True)
```

### Model Training

```python
from src.strategies.ml_models import train_model_on_processed_data

model, metrics = train_model_on_processed_data(
    processed_file='data/processed/processed_*.csv',
    ticker='AVGO',
    lookback_window=10,
    save_model_path='models/model_AVGO.pkl'
)
```

### Backtesting

```python
from run_backtest import run_backtest_ml

result = run_backtest_ml(
    processed_file='data/processed/processed_*.csv',
    ticker='AVGO',
    initial_cash=1000000.0,
    model_dir='models',
    min_confidence=0.6
)

report = result['report']
print(f"Total Return: {report['total_return']:.2f}%")
print(f"Sharpe Ratio: {report['sharpe_ratio']:.4f}")
```

## Directory Structure

```
LAB4-Algorithmic-Trading/
├── data/
│   ├── raw/                    # Raw market data
│   ├── processed/              # Processed data with features
│   └── selections/             # Stock selection files
├── checkpoints/                # Checkpoint files for resume
├── models/                     # Trained ML models
├── backtest_results/           # Backtest results and plots
├── src/
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── simulator.py           # Mock trading simulator
│   ├── evaluator.py           # Performance metrics
│   ├── stock_selector.py      # Stock selection from S&P 500
│   └── strategies/
│       ├── base_strategy.py   # Base strategy class
│       ├── technical_indicators.py  # Technical analysis indicators
│       └── ml_models.py       # XGBoost ML models
├── run_data_collection.py     # Data collection runner
├── run_ml_training.py         # ML model training script
├── run_backtest.py            # Backtesting script
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Performance Metrics

The system calculates comprehensive performance metrics:

- **Total Return**: Overall portfolio return percentage
- **Annualized Return**: Annualized return rate
- **Sharpe Ratio**: Risk-adjusted return metric
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Annualized standard deviation of returns
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss

## License

MIT — see [LICENSE](LICENSE).

---

*Originally developed as USC DSCI 560 coursework — refactored for clarity and portfolio presentation.*
