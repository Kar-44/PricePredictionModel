"""
Simple Price Prediction Model
=====================================
Predicts the NEXT trading day's closing price from historical sample data.
Datasets taken from: https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset

Approach
--------
- Feature engineering from price history: lagged closes, moving averages,
  volatility, volume change.
- Linear Regression as the "learning" model.
- Chronological (not random) train/test split, since this is time series data.
- Evaluation with MAE, RMSE, R^2.
- A chart comparing actual vs. predicted prices on the test set.

Usage
--------
    python3 SamplePred.py path/to/data.csv

Expects a CSV with columns: Date, Open, High, Low, Close, Adj Close, Volume
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(csv_path: str) -> pd.DataFrame:
    # Load csv making sure that correct columns are named properly and are in order
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Turn raw OHLCV rows into features a model can "learn" from.
    df = df.copy()

    # Daily return: % change from the previous close
    df["Return"] = df["Close"].pct_change()

    # Lagged closing prices: yesterday, 2 days ago, 3 days ago.
    # These let the model see recent price momentum.
    for lag in (1, 2, 3):
        df[f"Close_lag{lag}"] = df["Close"].shift(lag)

    # Moving averages smooth out day-to-day noise and capture trend.
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()

    # Rolling volatility (std dev of returns over 10 days) - captures risk.
    df["Volatility10"] = df["Return"].rolling(window=10).std()

    # Volume change - spikes in volume often precede price moves.
    df["Volume_change"] = df["Volume"].pct_change()

    # Target: TOMORROW's close. shift(-1) pulls the next row's value
    # up into the current row, which is exactly "what happens next".
    df["Target"] = df["Close"].shift(-1)

    # Rolling/shift operations create NaNs at the start (no history yet)
    # and end (no "next day" for the last row) - drop them.
    df = df.dropna().reset_index(drop=True)

    return df


def time_series_split(df, feature_cols, target_col, test_size=0.2):
    """
    Split chronologically: train on the earlier X%, test on the later
    (1-X)%. Never shuffle time series data - that would let the model
    "see the future" during training and give a falsely good score.
    """
    split_idx = int(len(df) * (1 - test_size))
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]
    dates_test = test["Date"]

    return X_train, X_test, y_train, y_test, dates_test


def train_model(X_train, y_train) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2:  {r2:.4f}")

    return predictions


def plot_results(dates_test, y_test, predictions, out_path="prediction_results.png"):
    y_test_vals = y_test.values
    errors = predictions - y_test_vals  # positive = overpredicted, negative = underpredicted

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Top panel: actual vs. predicted price
    ax1.plot(dates_test, y_test_vals, label="Actual Close", linewidth=2, color="blue")
    ax1.plot(dates_test, predictions, label="Predicted Close", linewidth=2, color="orange")
    ax1.set_title("Actual vs. Predicted Closing Price (test set)")
    ax1.set_ylabel("Price")
    ax1.legend()

    # Bottom panel: prediction error each day (0 = perfect prediction)
    ax2.bar(dates_test, errors, width=1.0, color=np.where(errors >= 0, "tab:red", "tab:green"))
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Prediction Error (Predicted - Actual)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Error")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nChart saved to {out_path}")
    plt.show()


def predict_next_close(model, df, feature_cols):
    #Use the most recent row of features to predict the next close.
    latest_features = df[feature_cols].iloc[[-1]]
    return model.predict(latest_features)[0]


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data.csv"

    df = load_data(csv_path)
    df = engineer_features(df)

    feature_cols = [
        "Close_lag1", "Close_lag2", "Close_lag3",
        "MA5", "MA10", "MA20",
        "Volatility10", "Volume_change",
    ]
    target_col = "Target"

    X_train, X_test, y_train, y_test, dates_test = time_series_split(
        df, feature_cols, target_col, test_size=0.2
    )

    model = train_model(X_train, y_train)
    predictions = evaluate_model(model, X_test, y_test)
    plot_results(dates_test, y_test, predictions)

    next_close = predict_next_close(model, df, feature_cols)
    print(f"\nPredicted next closing price: {next_close:.2f}")

    # Show which features the model weighted most heavily
    print("\nLearned coefficients (feature importance, roughly):")
    for name, coef in sorted(zip(feature_cols, model.coef_), key=lambda x: -abs(x[1])):
        print(f"  {name:<15} {coef:+.4f}")


if __name__ == "__main__":
    main()