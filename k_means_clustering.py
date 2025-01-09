import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('cleaned_data.csv', encoding='iso-8859-1', low_memory=False)

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Aggregate data by CustomerID
customer_data = df.groupby('CustomerID').agg({
    'Revenue': 'sum',                 # Total revenue per customer
    'InvoiceNo': 'count',             # Purchase frequency
    'Quantity': 'sum'                 # Total items bought
}).rename(columns={'InvoiceNo': 'Frequency', 'Quantity': 'TotalQuantity'})

# Reset index
customer_data = customer_data.reset_index()

# Scale the Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data[['Revenue', 'Frequency', 'TotalQuantity']])

# Determine Optimal Number of Clusters (Elbow Method)
inertia = []
range_clusters = range(1, 11)

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range_clusters, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Perform Clustering with Optimal K 
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(scaled_data)

# Evaluate Clusters
sil_score = silhouette_score(scaled_data, customer_data['Cluster'])
print(f"Silhouette Score: {sil_score}")

# Visualize Clusters
plt.figure(figsize=(10, 6))
plt.scatter(customer_data['Revenue'], customer_data['Frequency'], c=customer_data['Cluster'], cmap='viridis', alpha=0.6)
plt.title('Customer Clusters')
plt.xlabel('Revenue')
plt.ylabel('Frequency')
plt.colorbar(label='Cluster')
plt.show()
