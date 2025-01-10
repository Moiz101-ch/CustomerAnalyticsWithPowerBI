import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('cleaned_data.csv', encoding='iso-8859-1', low_memory=False)

print(df.head)

