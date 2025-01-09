import warnings
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Suppress warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('data.csv', encoding='iso-8859-1')

# Display the first few rows of the dataset
print("Initial Dataset Overview:\n", df.head())

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

# Drop rows with invalid InvoiceDate
df = df.dropna(subset=['InvoiceDate'])
print("\nDataset size after removing invalid dates:", df.shape)

# Validate the result
print(df['InvoiceDate'].dtypes)  # Should be datetime64[ns]

# Check for any parsing errors
print("Are there any nulls in 'InvoiceDate' after conversion?", df['InvoiceDate'].isnull().sum())

# Check for missing values
print("Missing values before handling:\n", df.isnull().sum())

# Check the percentage of missing values in 'CustomerID'
missing_customerid_pct = (df['CustomerID'].isnull().sum() / len(df)) * 100
print(f"Percentage of missing 'CustomerID' values: {missing_customerid_pct:.2f}%")

# Fill missing CustomerID with a placeholder value
df['CustomerID'] = df['CustomerID'].fillna(-1)

# Remove negative or zero Quantity and UnitPrice
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Remove duplicate rows
df = df.drop_duplicates()

# Create a revenue column
df['Revenue'] = df['Quantity'] * df['UnitPrice']

# Export data for Power BI
df.to_csv('cleaned_data.csv')
print("\nData exported for Power BI visualization")

# Aggregate data by time
time_series_data = df.resample('D', on='InvoiceDate').sum()[['Revenue']]

# print("\nAggregated time series data:")
print(time_series_data.head())

# Visualize the time series
plt.figure(figsize=(12, 6))
plt.plot(time_series_data.index, time_series_data['Revenue'], label='Daily Revenue', color='blue')
plt.title('Daily Revenue Over Time')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.show()

# Perform ADF Test to check stationarity
result = adfuller(time_series_data['Revenue'])
print("\nADF Test Results:")
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
print("Critical Values:")
for key, value in result[4].items():
    print(f"{key}: {value}")

# Interpretation
if result[1] <= 0.05:
    print("\nThe time series is stationary (p-value <= 0.05).")
else:
    print("\nThe time series is not stationary (p-value > 0.05). Differencing is needed.")

# Differencing the time series as it is not Stationary
time_series_data['Revenue_diff'] = time_series_data['Revenue'].diff().dropna()
time_series_data.dropna(subset=['Revenue_diff'], inplace=True)

# Check stationarity after differencing
result_diff = adfuller(time_series_data['Revenue_diff'].dropna())
print("\nADF Test Results After Differencing:")
print(f"ADF Statistic: {result_diff[0]}")
print(f"p-value: {result_diff[1]}")

if result_diff[1] <= 0.05:
    print("\nThe differenced time series is stationary (p-value <= 0.05).")
else:
    print("\nThe differenced time series is still not stationary (p-value > 0.05).")

# Decompose the time series
decomposition = seasonal_decompose(time_series_data['Revenue_diff'], model='additive', period=30)

# Plot the decomposition
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(decomposition.observed, label='Observed')
plt.legend(loc='upper left')

plt.subplot(412)
plt.plot(decomposition.trend, label='Trend', color='orange')
plt.legend(loc='upper left')

plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonality', color='green')
plt.legend(loc='upper left')

plt.subplot(414)
plt.plot(decomposition.resid, label='Residuals', color='red')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Plot ACF and PACF
plt.figure(figsize=(12, 6))

# ACF plot
plt.subplot(121)
plot_acf(time_series_data['Revenue'].dropna(), lags=40, ax=plt.gca())
plt.title("Autocorrelation Function (ACF)")

# PACF plot
plt.subplot(122)
plot_pacf(time_series_data['Revenue'].dropna(), lags=40, ax=plt.gca(), method='ywm')
plt.title("Partial Autocorrelation Function (PACF)")

# plt.tight_layout()
plt.show()

# Define the p, d, and q ranges
p = range(0, 5)  # Test values for AR terms
d = range(0, 2)  # Differencing order
q = range(0, 5)  # Test values for MA terms

# Generate all combinations of p, d, q
pdq_combinations = list(itertools.product(p, d, q))

# Initialize variables to store the best model and its AIC
best_aic = float("inf")
best_pdq = None
best_model = None

# Perform grid search
print("Performing Grid Search for ARIMA parameters...")

for pdq in pdq_combinations:
    try:
        # Fit ARIMA model
        model = ARIMA(time_series_data['Revenue_diff'], order=pdq)
        model_fit = model.fit()

        # Check AIC value
        aic = model_fit.aic
        print(f"ARIMA{pdq} - AIC: {aic}")

        # Update best model if current AIC is lower
        if aic < best_aic:
            best_aic = aic
            best_pdq = pdq
            best_model = model_fit

    except Exception as e:
        # Skip combinations that fail
        print(f"ARIMA{pdq} failed. Error: {e}")

# Output the best model and parameters
print("\nBest ARIMA Model:")
print(f"Order: {best_pdq}")
print(f"AIC: {best_aic}")

# Split data into train and test sets
train = time_series_data[:int(0.8 * len(time_series_data))]
test = time_series_data[int(0.8 * len(time_series_data)):]

# Fit ARIMA model using determined (p, d, q)
model = ARIMA(time_series_data['Revenue_diff'], order=(4, 0, 3)) 
arima_model = model.fit()

# Summary of the model
print("\nARIMA Model Summary:")
print(arima_model.summary())

# Forecast
forecast = arima_model.forecast(steps=len(test))

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Revenue_diff'], label='Train Data')
plt.plot(test.index, test['Revenue_diff'], label='Test Data', color='orange')
plt.plot(test.index, forecast, label='Forecast', color='green')
plt.title('ARIMA Model - Revenue Forecast')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.show()





