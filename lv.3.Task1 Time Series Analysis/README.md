Time Series Analysis: Air Passenger Forecasting

📌 Project Overview

This project focuses on analyzing, decomposing, smoothing, and forecasting historical time-series data using Python. Using the classic Air Passengers dataset, the project implements advanced statistical and machine learning modeling techniques to uncover underlying trends, seasonality, and generate accurate future predictions.



🛠️ Tech Stack \& Libraries



Language: Python



Environment: VS Code



**Core Libraries:**



* pandas \& numpy (Data manipulation and preprocessing)



* matplotlib (Data visualization and plotting)



* statsmodels (Time series decomposition, smoothing, and SARIMA modeling)



📊 Step-by-Step Project Workflow \& Implementation

**Step 1: Data Loading \& Inspection**

Action: Loaded the monthly passenger records from airpassengers.csv and set the Month column as a proper DatetimeIndex ranging from January 1949 to December 1960 (144 total entries with no missing values).



Visualization: Air Passenger Time Series.png — Illustrates the raw monthly passenger count, showcasing a clear upward trend and growing seasonal volatility over time.



**Step 2: Time Series Decomposition**

Action: Applied multiplicative seasonal decomposition (period=12) because the amplitude of seasonal variations increases proportionally with the upward trend.



Components Extracted:

* Trend: Steady, long-term upward trajectory from \~1949 to 1960.
* Seasonality: Consistent repeating annual cycles peaking during summer months.
* Residuals: Random noise remaining after removing trend and seasonality.
* Visualization: Time Series Decomposition of Air Passengers.png — Breaks down the original time series into its individual components.



**Step 3: Moving Average \& Smoothing Techniques**

Action: Applied smoothing algorithms to filter out short-term noise and highlight underlying patterns:

* 12-Month Moving Average: Smooths out seasonal spikes to reveal the core trend.
* Simple Exponential Smoothing (SES): Applies exponentially decreasing weights ($\\alpha = 0.2$).
* Holt-Winters Exponential Smoothing: Captures both additive trend and additive seasonality.
* Visualization: Time Series Smoothing Techniques.png — Compares the original trajectory against moving averages and smoothing models.



**Step 4: SARIMA Modeling \& Forecasting**

Action:



* Split the dataset into an 80% training set and a 20% test set.
* Fitted a Seasonal Autoregressive Integrated Moving Average (SARIMA) model with parameters order=(1, 1, 1) and seasonal\_order=(1, 1, 1, 12).
* Evaluated forecasting accuracy using Root Mean Squared Error (RMSE).
* Visualization: SARIMA Model - Time Series Forecasting.png — Contrasts training data, actual test data, and forecasted values.

