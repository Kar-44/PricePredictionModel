# Stock Price Prediction using Linear Regression

A simple yet effective stock price predictor that learns from historical OHLCV data to forecast next-day closing prices.

## Overview

This project uses a linear regression model to predict the next trading day's closing price from historical stock market data. It is designed as a transparent baseline that combines practical feature engineering with time-series-aware evaluation.

The model works with standard OHLCV data and focuses on interpretability, making it a good starting point for experimenting with financial forecasting pipelines.

## How It Works

### Feature engineering

The model creates predictive features from raw historical prices and volume data, including:

- Lagged price features.
- Moving averages over 5, 10, and 20 days.
- Rolling volatility measures.
- Volume-based signals.

These features help the model capture short-term momentum, recent trend behavior, and variability in price action.

### Time-series aware split

Instead of randomly splitting the dataset, the project uses a chronological train/test split. This prevents look-ahead bias and makes the evaluation more realistic for forecasting tasks.

### Linear regression baseline

Linear regression serves as an interpretable baseline model. Because its coefficients can be inspected directly, it is easier to understand which engineered features contribute most to the prediction.

### Visual evaluation

The project plots actual versus predicted prices and includes error visualization to make model performance easier to inspect beyond summary metrics alone.

## Usage

Run the script from the command line:

```bash
python price_prediction.py path/to/data.csv
```

### Expected input

The script expects a CSV file with the following columns:

- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Adj Close`
- `Volume`

## Output

After running the script, the project provides:

- Performance metrics: MAE, RMSE, and R².
- A side-by-side price chart with prediction error visualization.
- A feature importance ranking based on model coefficients.
- A next-day closing price forecast.

## Caveats

Linear regression can struggle with abrupt market moves and often lags during sharp price changes. Predictions should be compared against a simple naive baseline, such as assuming the next day's price will be unchanged, before being used for any serious decision-making.

## Future Improvements

Potential next steps for improving the project include:

- Ensemble models such as Random Forest or XGBoost to capture non-linear relationships.
- Walk-forward validation for more realistic performance tracking over time.
- Multi-stock training to improve generalization.
- A classification version that predicts price direction (up or down) instead of exact closing price.

## Notes

This project is intended as an educational and analytical baseline rather than a production-grade trading system. Market prices are noisy, non-stationary, and influenced by many external factors that a simple linear model cannot fully capture.
