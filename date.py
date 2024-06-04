import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt

# Load CSV file into a pandas DataFrame
df = pd.read_csv('output1.csv')  # Replace 'your_data.csv' with your actual data file

# Combine date and time into a single datetime column
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d-%m-%Y %I:%M %p')

# Select relevant features for clustering
spatial_features = ['latitude', 'longitude']
datetime_features = ['datetime']

# Standardize the spatial features
scaler_spatial = StandardScaler()
df[spatial_features] = scaler_spatial.fit_transform(df[spatial_features])

# Standardize the datetime feature
scaler_datetime = StandardScaler()
df[datetime_features] = scaler_datetime.fit_transform(df[datetime_features])

# Perform KMeans clustering for spatial features
k_spatial = 3  # You can adjust this value based on the elbow method or your preferences
kmeans_spatial = KMeans(n_clusters=k_spatial, random_state=42)
df['cluster_spatial'] = kmeans_spatial.fit_predict(df[spatial_features])

# Perform KMeans clustering for datetime features
k_datetime = 3  # You can adjust this value based on the elbow method or your preferences
kmeans_datetime = KMeans(n_clusters=k_datetime, random_state=42)
df['cluster_datetime'] = kmeans_datetime.fit_predict(df[datetime_features])

# Visualize the clusters with a scatter plot (spatial)
plt.scatter(df['latitude'], df['longitude'], c=df['cluster_spatial'], cmap='viridis', marker='o', s=50, label='Data Points')
plt.scatter(kmeans_spatial.cluster_centers_[:, 0], kmeans_spatial.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Clusters based on Spatial Coordinates (KMeans)')
plt.legend()
plt.show()

# Visualize the clusters with a scatter plot (datetime)
plt.scatter(df['datetime'], [0] * len(df), c=df['cluster_datetime'], cmap='viridis', marker='o', s=50, label='Data Points')
plt.scatter(kmeans_datetime.cluster_centers_[:, 0], [0] * k_datetime, c='red', marker='X', s=200, label='Centroids')
plt.xlabel('Datetime')
plt.title('Clusters based on Datetime (KMeans)')
plt.legend()
plt.show()
