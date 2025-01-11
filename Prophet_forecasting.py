import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.graph_objs as go

# Load dataset
time_series_data = pd.read_csv('cleaned_data.csv', encoding='iso-8859-1', low_memory=False)

# Convert InvoiceDate to datetime format
time_series_data['InvoiceDate'] = pd.to_datetime(time_series_data['InvoiceDate'])

# Group by date and sum the revenue for each day
daily_data = time_series_data.groupby(time_series_data['InvoiceDate'].dt.date)['Revenue'].sum().reset_index()

# Rename columns to fit Prophet's expected format
daily_data.rename(columns={'InvoiceDate': 'ds', 'Revenue': 'y'}, inplace=True)

# Handle missing values
# Prophet will automatically handle missing values in the time series, but let's ensure there are no gaps
daily_data = daily_data.set_index('ds').asfreq('D', fill_value=0).reset_index()
daily_data = daily_data [['ds', 'y']]
mean_value = daily_data['y'][daily_data['y'] > 0].mean()  # Exclude zeros from mean calculation
daily_data['y'] = daily_data['y'].replace(0, mean_value)  # Replace zeros with the calculated mean

# Step 6: Split data into training and testing sets
train = daily_data[daily_data['ds'] < '2011-01-01']  # Use data before 2011 for training
test = daily_data[daily_data['ds'] >= '2011-01-01']  # Use data after 2011 for testing

# Step 7: Initialize and fit the Prophet model
model = Prophet()  # Enable daily seasonality for high-frequency data
model.add_country_holidays(country_name='UK')  # Add UK holidays
model.fit(train)

# Make future dataframe for forecasting
future = model.make_future_dataframe(periods = 30)  # Forecast for 1 year ahead
forecast = model.predict(future)

# Visualize the forecast
fig = model.plot(forecast)
plt.title("Revenue Forecast with Prophet")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.show()

# Evaluate Model Performance
# Compare the forecast with actual values in the test set
forecasted_values = forecast[['ds', 'yhat']].tail(len(test))
actual_values = test[['ds', 'y']]
print(f'Actual Values: {actual_values}')

# Merge predicted and actual values for evaluation
comparison_df = pd.merge(forecasted_values, actual_values, on='ds', how='left')
comparison_df['error'] = comparison_df['y'] - comparison_df['yhat']

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(np.mean(comparison_df['error']**2))
print(f'RMSE: {rmse}')

# Interactive plot using Plotly for better visualization
trace1 = go.Scatter(x=comparison_df['ds'], y=comparison_df['y'], mode='lines', name='Actual')
trace2 = go.Scatter(x=comparison_df['ds'], y=comparison_df['yhat'], mode='lines', name='Predicted')

layout = go.Layout(title="Actual vs Predicted Revenue", xaxis={'title': 'Date'}, yaxis={'title': 'Revenue'})
fig = go.Figure(data=[trace1, trace2], layout=layout)
fig.show()