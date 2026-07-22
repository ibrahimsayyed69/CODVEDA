from pkgutil import get_data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
# ==========================================
# STEP 1: Load and Inspect Dataset
# ==========================================
#1. load dataset
df = pd.read_csv("airpassengers.csv")
#2. Converting the 'month' column to datetime and set it as the index
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
#3. Display basic info
print("Dataset Head:")
print(df.head())
print("\nDataset Info:")
print(df.info())
#4. Plot the raw time series
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["#Passengers"], color="tab:blue", linewidth=2)
plt.title("Air passengers Time Series", fontsize=14)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Number of Passengers", fontsize=12)
plt.grid(True)
plt.show()
# ==========================================
# STEP 2: Time Series Decomposition
# ==========================================
#1. personel seasonal decomposition
decomposition = seasonal_decompose(df["#Passengers"], model="multiplicative", period=12)
#2. Extract Components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
#3. plot the decomposed components
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
axes[0].plot(df["#Passengers"], label="Original", color="tab:blue")
axes[0].legend(loc="upper left")
axes[0].set_ylabel("Original")
#just a break for axes[1]
axes[1].plot(trend, label="Trend", color="tab:orange")
axes[1].legend(loc="upper left")
axes[1].set_ylabel("Trend")
#just a little break for axes[2]
axes[2].plot(seasonal, label="Seasonality", color="tab:green")
axes[2].legend(loc="upper left")
axes[2].set_ylabel("Seasonal")
#just a little break for axes[2]
axes[3].plot(residual, label="Residual", color="tab:red")
axes[3].legend(loc="upper left")
axes[3].set_ylabel("Residual")
plt.suptitle(
    "Time Series Decomposition of Air Passengers", fontsize=16, y=0.92
)
plt.tight_layout()
plt.show()
# ==========================================
# STEP 3: Moving Average & Smoothing Techniques
# ==========================================
#1. moving averge smoothing
df["Moving_Avg"] = df["#Passengers"].rolling(window=12).mean()
#2. simple Expo smoothing(ses)
df["SES"] = SimpleExpSmoothing(df["#Passengers"]).fit(
    smoothing_level=0.2, optimized=False
).fittedvalues
#3. Holt-winters
df["Holt_Winters"] = (
    ExponentialSmoothing(
        df["#Passengers"],trend="add",
        seasonal="add",
        seasonal_periods=12,
   )
    .fit()
    .fittedvalues
)
#4. plot smoothing
plt.figure(figsize=(12, 6))
plt.plot(
    df["#Passengers"],
    label="Original",
    color="tab:blue",
    alpha=0.5,
    linewidth=1.5,
)
plt.plot(
    df["Moving_Avg"],
    label="12-Month Moving Average",
    color="tab:orange",
    linewidth=2,
)
plt.plot(
    df["SES"],
    label="Simpe Expo Smoothing (aplha=0.2)",
    color="tab:green",
    linestyle="--",
)
plt.plot(
    df["Holt_Winters"],
    label="Holt-Winters (Trend + Seasonal)",
    color="tab:red",
    linewidth=2,
)

plt.title("Time Series Smoothing Techniques", fontsize=14)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Number of Passengers", fontsize=12)
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show
# ==========================================
# STEP 4: Build SARIMA Model & Forecasting
# ==========================================
#1. split data into traina and test
train_size = int(len(df) * 0.8)
train_data = df["#Passengers"][:train_size]
test_data = df["#Passengers"][train_size:]
#2. fit SARIMA model
model = SARIMAX(
    train_data,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False,
)
sarima_fit = model.fit(disp=False)
#3. forecast for test
predictions = sarima_fit.get_forecast(steps=len(test_data)).predicted_mean
#4. RMSE
rmse = root_mean_squared_error(test_data, predictions)
print("f\nSARIMA Model Test RMSE: {rmse:.2f}")
#5. Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data, label="train Data", color="tab:blue")
plt.plot(
    test_data.index, test_data, label="Actual Test Data", color="tab:orange"
)
plt.plot(
    predictions.index,
    predictions,
    label="tab:green",
    linestyle="--",
)

plt.title("SARIMA Model - Time Series Forecasting", fontsize=14)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Number of Passengers", fontsize=12)
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()