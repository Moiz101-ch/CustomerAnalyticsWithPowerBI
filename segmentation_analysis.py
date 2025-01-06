import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
data = pd.read_csv('data.csv', encoding='iso-8859-1')

# Checking missing value in each column
missing_values = data.isnull().sum()
#print(missing_values)

# Dropping rows with missing values in CustomerID
data.dropna(subset=['CustomerID'], inplace=True)

# Count negative values in the 'Quantity' column
negative_quantity_count = (data['Quantity'] < 0).sum()

# Count negative values in the 'UnitPrice' column
negative_unit_price_count = (data['UnitPrice'] < 0).sum()

# print(f"Negative values in 'Quantity': {negative_quantity_count}")
# print(f"Negative values in 'UnitPrice': {negative_unit_price_count}")

# Removing rows with negative values in 'Quantity' & 'UnitPrice' columns
data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]

# Checking datatypes of each column
#print(data.dtypes)

# Converting CustomerID to type integer
data['CustomerID'] = data['CustomerID'].astype(int)

# Structuring date by time
data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
data.set_index('InvoiceDate', inplace=True)

# Checking for duplicate dates
duplicate_dates = data.index.duplicated().sum()

# print(f"Number of duplicate dates: {duplicate_dates}")

data = data.groupby(data.index).agg({'Quantity': 'sum', 'UnitPrice': 'mean'})  # Customize aggregation

# Now, setting the frequency to daily
data = data.asfreq('D')

# print(f"Daily Frequency: \n{data}\n")

# Interpolating for any gaps
data.interpolate(method='linear', inplace=True)

# Checking the result
# print(data.head())

data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]

# Grouping data by day and calculating the sum of revenue for each day
time_series_data = data.groupby(data.index).agg({"TotalPrice": "sum"})

time_series_data = time_series_data.rename(columns={"TotalPrice": "Revenue"})

# Plot the time series
# plt.figure(figsize=(12, 6))
# plt.plot(time_series_data, label="Revenue")
# plt.title("Daily Revenue Over Time")
# plt.xlabel("Date")
# plt.ylabel("Revenue")
# plt.legend()
# plt.show()

# Checking stationarity
# adf_test = adfuller(time_series_data["Revenue"])

# if adf_test[1] < 0.05:
#     print(f"The time series is stationary: {adf_test[1]}")
# else:
#     print(f"The time series is not stationary: {adf_test[1]}")

# Plotting ACF and PACF
# plt.figure(figsize=(12, 6))
# plt.subplot(211)
# plot_acf(time_series_data["Revenue"], ax=plt.gca(), lags=25)
# plt.subplot(212)
# plot_pacf(time_series_data["Revenue"], ax=plt.gca(), lags=25)
# plt.show()

# Seasonal decomposition
# result = seasonal_decompose(time_series_data["Revenue"], model='additive', period = 30)
# result.plot()
# plt.show()

# ARIMA model fitting
p, d, q = 1, 0, 1 
model = ARIMA(time_series_data["Revenue"], order=(p, d, q))
model_fit = model.fit()

# Model summary
print(model_fit.summary())













