from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


st.set_page_config(page_title="Stock Market Predictor", page_icon="📈", layout="wide")

MODEL_PATH = Path(__file__).with_name("Stock Predictions Model.keras")
WINDOW = 100
FORECAST_DAYS = 60
CHART_HISTORY_DAYS = 400
PREDICTION_DAYS = 252


@st.cache_resource
def load_prediction_model(model_path):
    from keras.models import load_model

    return load_model(model_path)


@st.cache_data(ttl=900)
def download_prices(symbol, start_date, end_date):
    prices = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    if "Close" not in prices.columns:
        return pd.DataFrame()
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    return prices.dropna(subset=["Close"])


@st.cache_data(ttl=900)
def download_hourly_prices(symbol):
    prices = yf.download(
        symbol,
        period="60d",
        interval="1h",
        auto_adjust=False,
        progress=False,
    )
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    required_columns = {"High", "Low", "Close"}
    if not required_columns.issubset(prices.columns):
        return pd.DataFrame()
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    return prices.dropna(subset=list(required_columns))


def make_prediction(prices, model, prediction_days=PREDICTION_DAYS):
    if len(prices) < WINDOW + prediction_days:
        raise ValueError(f"At least {WINDOW + prediction_days} daily rows are required.")
    test = prices[["Close"]].tail(prediction_days)
    context_end = len(prices) - prediction_days
    context = prices[["Close"]].iloc[context_end - WINDOW:context_end]
    test_with_context = pd.concat([context, test], ignore_index=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(test_with_context)
    sequences = np.array([scaled[i - WINDOW:i] for i in range(WINDOW, len(scaled))])
    actual = test_with_context.iloc[WINDOW:]["Close"].to_numpy()
    predicted = scaler.inverse_transform(model.predict(sequences, verbose=0)).ravel()
    return pd.DataFrame({"Actual": actual, "Predicted": predicted}, index=test.index)


def make_future_prediction(prices, model, days=FORECAST_DAYS):
    close = prices[["Close"]].astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(close)
    sequence = scaled_close[-WINDOW:].reshape(1, WINDOW, 1)
    predictions = []

    for _ in range(days):
        next_scaled = float(model.predict(sequence, verbose=0).reshape(-1)[0])
        predictions.append(next_scaled)
        sequence = np.concatenate(
            [sequence[:, 1:, :], np.array(next_scaled).reshape(1, 1, 1)], axis=1
        )

    predicted_close = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
    future_dates = pd.bdate_range(
        prices.index[-1] + pd.Timedelta(days=1), periods=days
    )
    average_range = ((prices["High"] - prices["Low"]) / prices["Close"]).median()
    average_range = float(average_range) if pd.notna(average_range) else 0.02
    return pd.DataFrame(
        {
            "High": predicted_close * (1 + average_range / 2),
            "Low": predicted_close * (1 - average_range / 2),
            "Close": predicted_close,
            "Predicted": predicted_close,
        },
        index=future_dates,
    )


def expected_signal(current_price, forecast_price):
    change = (forecast_price - current_price) / current_price * 100
    if change >= 1:
        return "Buy"
    if change <= -1:
        return "Sell"
    return "Hold"


def make_hourly_forecast(prices, hours=24, window=24):
    close = prices["Close"].astype(float).tail(500)
    if len(close) < window + 20:
        raise ValueError("Not enough hourly data is available for a swing forecast.")

    values = close.to_numpy()
    features = np.array([values[index - window:index] for index in range(window, len(values))])
    targets = values[window:]
    model = LinearRegression().fit(features, targets)
    sequence = values[-window:].copy()
    predictions = []
    for _ in range(hours):
        prediction = float(model.predict(sequence.reshape(1, -1))[0])
        predictions.append(prediction)
        sequence = np.concatenate([sequence[1:], [prediction]])

    future_index = pd.date_range(
        close.index[-1] + pd.Timedelta(hours=1), periods=hours, freq="h"
    )
    typical_range = ((prices["High"] - prices["Low"]) / prices["Close"]).median()
    typical_range = float(typical_range) if pd.notna(typical_range) else 0.01
    return pd.DataFrame(
        {
            "High": np.array(predictions) * (1 + typical_range / 2),
            "Low": np.array(predictions) * (1 - typical_range / 2),
            "Close": predictions,
            "Prediction": predictions,
        },
        index=future_index,
    )


def calculate_supertrend(prices, period=10, multiplier=3.0):
    high = prices["High"].astype(float)
    low = prices["Low"].astype(float)
    close = prices["Close"].astype(float)
    previous_close = close.shift(1)

    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    average_true_range = true_range.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    midpoint = (high + low) / 2
    basic_upper = midpoint + multiplier * average_true_range
    basic_lower = midpoint - multiplier * average_true_range

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    trend = pd.Series(1, index=prices.index, dtype=int)
    supertrend = pd.Series(np.nan, index=prices.index, dtype=float)

    for index in range(1, len(prices)):
        if pd.isna(basic_upper.iloc[index - 1]):
            continue
        if (
            basic_upper.iloc[index] < final_upper.iloc[index - 1]
            or close.iloc[index - 1] > final_upper.iloc[index - 1]
        ):
            final_upper.iloc[index] = basic_upper.iloc[index]
        else:
            final_upper.iloc[index] = final_upper.iloc[index - 1]
        if (
            basic_lower.iloc[index] > final_lower.iloc[index - 1]
            or close.iloc[index - 1] < final_lower.iloc[index - 1]
        ):
            final_lower.iloc[index] = basic_lower.iloc[index]
        else:
            final_lower.iloc[index] = final_lower.iloc[index - 1]

        if close.iloc[index] > final_upper.iloc[index - 1]:
            trend.iloc[index] = 1
        elif close.iloc[index] < final_lower.iloc[index - 1]:
            trend.iloc[index] = -1
        else:
            trend.iloc[index] = trend.iloc[index - 1]
        supertrend.iloc[index] = (
            final_lower.iloc[index] if trend.iloc[index] == 1 else final_upper.iloc[index]
        )

    signal = trend.diff().map({2: "Buy", -2: "Sell"})
    return pd.DataFrame(
        {"Supertrend": supertrend, "Trend": trend, "Signal": signal},
        index=prices.index,
    )


st.title("Stock Price Prediction")
st.caption("Analyze historical prices and compare the trained model with original closing prices.")

with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("Stock symbol", "GOOG").strip().upper()
    start_date = st.date_input("Start date", pd.Timestamp("2012-01-01"))
    end_date = st.date_input("End date", pd.Timestamp.today())
    run_prediction = st.button("Run prediction", type="primary", use_container_width=True)

if not symbol:
    st.info("Enter a stock symbol to begin.")
    st.stop()
if not re.fullmatch(r"[A-Z0-9.^=-]{1,15}", symbol):
    st.error("Enter a valid stock ticker, for example `GOOG`, `AAPL`, or `^GSPC`.")
    st.stop()
if start_date >= end_date:
    st.error("The end date must be after the start date.")
    st.stop()

try:
    prices = download_prices(symbol, start_date, end_date)
except Exception as error:
    st.error(f"Could not download data for {symbol}: {error}")
    st.stop()

if prices.empty:
    st.error(f"No price data was returned for {symbol}. Check the symbol and date range.")
    st.stop()
if not {"High", "Low", "Close"}.issubset(prices.columns):
    st.error(f"Complete high, low, and close data is unavailable for {symbol}.")
    st.stop()
if len(prices) < WINDOW + 20:
    st.error("Choose a wider date range. At least 120 trading days are needed.")
    st.stop()

st.subheader(f"{symbol} historical prices")
st.line_chart(prices["Close"], y_label="Closing price")

ma_20 = prices["Close"].rolling(20).mean()
ma_50 = prices["Close"].rolling(50).mean()
ma_100 = prices["Close"].rolling(100).mean()
ma_200 = prices["Close"].rolling(200).mean()

def plot_lines(title, lines):
    st.subheader(title)
    figure, axis = plt.subplots(figsize=(12, 4))
    for label, series, color in lines:
        axis.plot(series.index, series, label=label, color=color)
    axis.set_ylabel("Price")
    axis.legend()
    axis.grid(alpha=0.2)
    st.pyplot(figure, clear_figure=True)


ma_cutoff = prices.index.max() - pd.DateOffset(years=1)
ma_chart = prices.loc[prices.index >= ma_cutoff].copy()
ma_20_chart = ma_20.loc[ma_chart.index]
ma_50_chart = ma_50.loc[ma_chart.index]
buy_signal = (ma_20 > ma_50) & (ma_20.shift(1) <= ma_50.shift(1))
sell_signal = (ma_20 < ma_50) & (ma_20.shift(1) >= ma_50.shift(1))
buy_dates = ma_chart.index[buy_signal.loc[ma_chart.index].fillna(False)]
sell_dates = ma_chart.index[sell_signal.loc[ma_chart.index].fillna(False)]

st.subheader("Close Price vs 20 MA vs 50 MA - Last 1 Year")
ma_figure, ma_axis = plt.subplots(figsize=(14, 6))
ma_axis.plot(ma_chart.index, ma_chart["Close"], label="Close price", color="#16324F", linewidth=1.3)
ma_axis.plot(ma_chart.index, ma_20_chart, label="20 MA", color="#FFD700", linewidth=1.8)
ma_axis.plot(ma_chart.index, ma_50_chart, label="50 MA", color="#E45756", linewidth=1.8)
ma_axis.scatter(
    buy_dates,
    ma_chart.loc[buy_dates, "Close"],
    marker="^",
    color="#16A085",
    s=90,
    label="Buy: 20 MA crossed above 50 MA",
    zorder=5,
)
ma_axis.scatter(
    sell_dates,
    ma_chart.loc[sell_dates, "Close"],
    marker="v",
    color="#C0392B",
    s=90,
    label="Sell: 20 MA crossed below 50 MA",
    zorder=5,
)
ma_axis.set_ylabel("Price")
ma_axis.set_xlim(ma_chart.index[0], ma_chart.index[-1])
ma_axis.legend()
ma_axis.grid(alpha=0.2)
ma_figure.autofmt_xdate()
st.pyplot(ma_figure, clear_figure=True)
plot_lines(
    "Close Price vs 100 MA",
    [("Close price", prices["Close"], "#16324F"), ("100 MA", ma_100, "#E45756")],
)
plot_lines(
    "Close Price vs 200 MA",
    [("Close price", prices["Close"], "#16324F"), ("200 MA", ma_200, "#2A9D8F")],
)
plot_lines(
    "100 MA vs 200 MA",
    [("100 MA", ma_100, "#E45756"), ("200 MA", ma_200, "#2A9D8F")],
)

supertrend = calculate_supertrend(prices, period=15, multiplier=1.5)
result = None
future_prices = None

if run_prediction:
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found: `{MODEL_PATH.name}`. Run the notebook through its "
            "final `model.save(...)` cell, then click Run prediction again."
        )
    else:
        try:
            with st.spinner("Loading model and generating predictions..."):
                model = load_prediction_model(str(MODEL_PATH))
                result = make_prediction(prices, model, prediction_days=PREDICTION_DAYS)
                future_prices = make_future_prediction(prices, model)
        except Exception as error:
            st.error(f"The prediction could not be generated for {symbol}: {error}")

    if result is not None:
        current_close = float(prices["Close"].iloc[-1])
        latest_prediction = float(future_prices["Predicted"].iloc[0])
        change = (latest_prediction - current_close) / current_close * 100
        first, second, third = st.columns(3)
        first.metric("Latest close", f"₹{current_close:,.2f}")
        second.metric("Next-day model prediction", f"₹{latest_prediction:,.2f}")
        third.metric("Predicted change", f"{change:+.2f}%")

        moving_averages = {
            "20 MA": prices["Close"].rolling(20).mean().iloc[-1],
            "50 MA": prices["Close"].rolling(50).mean().iloc[-1],
            "100 MA": prices["Close"].rolling(100).mean().iloc[-1],
            "200 MA": prices["Close"].rolling(200).mean().iloc[-1],
        }
        current_trend = supertrend["Trend"].iloc[-1]
        current_supertrend = supertrend["Supertrend"].iloc[-1]
        supertrend_direction = "Bullish" if current_trend == 1 else "Bearish"
        ma_values = list(moving_averages.values())
        if all(left > right for left, right in zip(ma_values, ma_values[1:])):
            moving_average_direction = "Bullish"
        elif all(left < right for left, right in zip(ma_values, ma_values[1:])):
            moving_average_direction = "Bearish"
        else:
            moving_average_direction = "Mixed"
        if (
            current_close > moving_averages["20 MA"] > moving_averages["50 MA"]
            and current_trend == 1
        ):
            technical_suggestion = "Buy"
        elif (
            current_close < moving_averages["20 MA"] < moving_averages["50 MA"]
            and current_trend == -1
        ):
            technical_suggestion = "Sell"
        else:
            technical_suggestion = "Hold"
        short_prediction = float(future_prices["Predicted"].iloc[14])
        long_prediction = float(future_prices["Predicted"].iloc[59])
        short_signal = expected_signal(current_close, short_prediction)
        long_signal = expected_signal(current_close, long_prediction)
        if short_signal == long_signal == technical_suggestion:
            model_technical_suggestion = short_signal
        elif short_signal == long_signal and short_signal in {"Buy", "Sell"}:
            model_technical_suggestion = f"{short_signal} (model consensus)"
        else:
            model_technical_suggestion = "Hold (mixed signals)"
        summary = pd.DataFrame(
            {
                "Metric": [
                    "Latest close",
                    "Next-day model prediction",
                    "Predicted change",
                    "20 MA",
                    "50 MA",
                    "100 MA",
                    "200 MA",
                    "Moving average direction",
                    "Supertrend (15, 1.5)",
                    "Supertrend direction",
                    "Model short-term prediction (10-15 days)",
                    "Model long-term prediction (40-60 days)",
                    "Short-term expected signal",
                    "Long-term expected signal",
                    "Model technical suggestion",
                ],
                "Value": [
                    f"₹{current_close:,.2f}",
                    f"₹{latest_prediction:,.2f}",
                    f"{change:+.2f}%",
                    *[f"₹{value:,.2f}" for value in moving_averages.values()],
                    moving_average_direction,
                    f"₹{current_supertrend:,.2f}",
                    supertrend_direction,
                    f"₹{short_prediction:,.2f}",
                    f"₹{long_prediction:,.2f}",
                    short_signal,
                    long_signal,
                    model_technical_suggestion,
                ],
            }
        )
        st.subheader("Prediction Summary")
        st.table(summary)

        st.subheader("Prediction vs Original - Last 1 Year")
        st.line_chart(
            result.rename(columns={"Actual": "Original", "Predicted": "Prediction"}),
            y_label="Price",
        )
else:
    st.info("Enter a stock ticker and click Run prediction to compare the model output.")

if future_prices is not None:
    chart_prices = pd.concat([prices[["High", "Low", "Close"]], future_prices])
    chart_supertrend = calculate_supertrend(chart_prices, period=15, multiplier=1.5)
else:
    chart_prices = prices
    chart_supertrend = supertrend

chart_start = max(0, len(prices) - CHART_HISTORY_DAYS)
chart_prices = chart_prices.iloc[chart_start:]
chart_supertrend = chart_supertrend.loc[chart_prices.index]
st.subheader("Long-Term Close Price vs Prediction with Supertrend Signals (15, 1.5)")
figure, axis = plt.subplots(figsize=(14, 6))
axis.plot(
    chart_prices.index,
    chart_prices["Close"],
    label="Close price",
    color="#16324F",
    linewidth=1.3,
)
if result is not None:
    historical_prediction = result["Predicted"].loc[
        result.index.intersection(chart_prices.index)
    ]
    axis.plot(
        historical_prediction.index,
        historical_prediction,
        label="Historical prediction",
        color="#E45756",
        linewidth=1.4,
    )
    axis.plot(
        future_prices.index,
        future_prices["Predicted"],
        label="60-day forecast",
        color="#FF7F0E",
        linewidth=2,
        linestyle="--",
    )
axis.plot(
    chart_supertrend.index,
    chart_supertrend["Supertrend"],
    label="Supertrend (15, 1.5)",
    color="#8E44AD",
    linewidth=1.5,
)
buy_dates = chart_supertrend.index[chart_supertrend["Signal"] == "Buy"]
sell_dates = chart_supertrend.index[chart_supertrend["Signal"] == "Sell"]
axis.scatter(
    buy_dates,
    chart_prices.loc[buy_dates, "Close"],
    marker="^",
    color="#16A085",
    s=70,
    label="Buy",
    zorder=5,
)
axis.scatter(
    sell_dates,
    chart_prices.loc[sell_dates, "Close"],
    marker="v",
    color="#C0392B",
    s=70,
    label="Sell",
    zorder=5,
)
axis.set_ylabel("Price")
axis.set_xlim(chart_prices.index[0], chart_prices.index[-1])
axis.legend()
axis.grid(alpha=0.2)
st.pyplot(figure, clear_figure=True)
if future_prices is not None:
    future_supertrend = chart_supertrend.loc[future_prices.index]
    future_signals = future_supertrend
    future_signals = future_signals[future_signals["Signal"].notna()]
    if future_signals.empty:
        st.info("No Supertrend Buy/Sell change was detected in the next 60 forecast days.")
    else:
        st.subheader("Projected signals for the next 60 business days")
        st.dataframe(
            future_signals[["Signal", "Supertrend"]]
            .join(future_prices["Predicted"].rename("Forecast price"))
            .rename_axis("Date"),
            use_container_width=True,
        )

if run_prediction:
    try:
        hourly_prices = download_hourly_prices(symbol)
        if hourly_prices.empty:
            st.warning(f"Hourly data is not available for {symbol}.")
        else:
            hourly_forecast = make_hourly_forecast(hourly_prices)
            hourly_combined = pd.concat(
                [hourly_prices[["High", "Low", "Close"]], hourly_forecast]
            )
            hourly_supertrend = calculate_supertrend(
                hourly_combined, period=15, multiplier=1.5
            )
            hourly_history_supertrend = calculate_supertrend(
                hourly_prices, period=15, multiplier=1.5
            )
            hourly_history = hourly_prices.tail(24 * 10)
            hourly_chart_index = hourly_history.index.union(hourly_forecast.index)
            hourly_chart_prices = hourly_combined.loc[hourly_chart_index]
            hourly_chart_supertrend = hourly_supertrend.loc[hourly_chart_index]

            st.subheader("Hourly Swing Movement: Last 10 Days + Next 24 Hours")
            st.caption(
                "Hourly forecast uses a rolling regression model. It is separate from the daily LSTM model."
            )
            hourly_figure, hourly_axis = plt.subplots(figsize=(14, 6))
            hourly_axis.plot(
                hourly_history.index,
                hourly_history["Close"],
                label="Hourly close",
                color="#16324F",
                linewidth=1.2,
            )
            hourly_axis.plot(
                hourly_forecast.index,
                hourly_forecast["Prediction"],
                label="24-hour prediction",
                color="#FF7F0E",
                linestyle="--",
                linewidth=2,
            )
            hourly_axis.plot(
                hourly_chart_supertrend.index,
                hourly_chart_supertrend["Supertrend"],
                label="Supertrend (15, 1.5)",
                color="#8E44AD",
                linewidth=1.4,
            )
            hourly_buy_dates = hourly_history_supertrend.index[
                hourly_history_supertrend["Signal"] == "Buy"
            ].intersection(hourly_chart_index)
            hourly_sell_dates = hourly_history_supertrend.index[
                hourly_history_supertrend["Signal"] == "Sell"
            ].intersection(hourly_chart_index)
            hourly_axis.scatter(
                hourly_buy_dates,
                hourly_chart_prices.loc[hourly_buy_dates, "Close"],
                marker="^",
                color="#16A085",
                s=65,
                label="Historical Buy",
                zorder=5,
            )
            hourly_axis.scatter(
                hourly_sell_dates,
                hourly_chart_prices.loc[hourly_sell_dates, "Close"],
                marker="v",
                color="#C0392B",
                s=65,
                label="Historical Sell",
                zorder=5,
            )
            hourly_axis.set_ylabel("Price")
            hourly_axis.set_xlabel("Hourly timestamp")
            hourly_axis.legend()
            hourly_axis.grid(alpha=0.2)
            hourly_figure.autofmt_xdate()
            st.pyplot(hourly_figure, clear_figure=True)

            current_hourly_close = float(hourly_prices["Close"].iloc[-1])
            hourly_ema_20 = hourly_prices["Close"].ewm(span=20, adjust=False).mean().iloc[-1]
            hourly_ema_50 = hourly_prices["Close"].ewm(span=50, adjust=False).mean().iloc[-1]
            hourly_previous_close = hourly_prices["Close"].shift(1)
            hourly_true_range = pd.concat(
                [
                    hourly_prices["High"] - hourly_prices["Low"],
                    (hourly_prices["High"] - hourly_previous_close).abs(),
                    (hourly_prices["Low"] - hourly_previous_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            hourly_atr = float(hourly_true_range.ewm(span=14, adjust=False).mean().iloc[-1])
            hourly_trend = int(hourly_history_supertrend["Trend"].iloc[-1])
            latest_hourly_signal = hourly_history_supertrend["Signal"].iloc[-1]
            hourly_forecast_end = float(hourly_forecast["Prediction"].iloc[-1])
            hourly_forecast_change = (
                (hourly_forecast_end - current_hourly_close) / current_hourly_close * 100
            )
            hourly_bullish = (
                hourly_trend == 1
                and current_hourly_close > hourly_ema_20 > hourly_ema_50
            )
            hourly_bearish = (
                hourly_trend == -1
                and current_hourly_close < hourly_ema_20 < hourly_ema_50
            )
            if hourly_forecast_change >= 1 and hourly_bullish:
                hourly_signal = "Buy"
                hourly_reason = "Forecast is positive; price is above aligned EMA 20/50 and Supertrend is bullish."
                stop_loss = current_hourly_close - 1.5 * hourly_atr
                target = current_hourly_close + 3 * hourly_atr
                holding_period = "4-24 hours"
            elif hourly_forecast_change <= -1 and hourly_bearish:
                hourly_signal = "Sell"
                hourly_reason = "Forecast is negative; price is below aligned EMA 20/50 and Supertrend is bearish."
                stop_loss = current_hourly_close + 1.5 * hourly_atr
                target = current_hourly_close - 3 * hourly_atr
                holding_period = "4-24 hours"
            else:
                hourly_signal = "Hold"
                hourly_reason = "Forecast and technical indicators are not sufficiently aligned for a trade."
                stop_loss = np.nan
                target = np.nan
                holding_period = "Wait and reassess next hourly candle"

            hourly_summary = pd.DataFrame(
                {
                    "Item": [
                        "Hourly signal",
                        "Supertrend direction",
                        "Latest Supertrend crossover",
                        "Model short-term prediction (24 hours)",
                        "24-hour forecast direction",
                        "Swing expected signal (next 1 day)",
                    ],
                    "Value": [
                        hourly_signal,
                        "Bullish" if hourly_trend == 1 else "Bearish",
                        latest_hourly_signal if pd.notna(latest_hourly_signal) else "No new crossover",
                        f"₹{hourly_forecast_end:,.2f}",
                        "Upward" if hourly_forecast_change > 0 else "Downward",
                        hourly_signal,
                    ],
                }
            )
            st.subheader("Hourly Swing Technical Summary")
            st.table(hourly_summary)
    except Exception as error:
        st.warning(f"Hourly swing chart could not be generated: {error}")
st.caption("This is an educational model output, not financial advice.")