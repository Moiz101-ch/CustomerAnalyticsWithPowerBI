import pandas as pd
import matplotlib.pyplot as plt

# Alternative encoding
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






