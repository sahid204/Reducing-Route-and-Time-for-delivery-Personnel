import pandas as pd
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load CSV file into a pandas DataFrame
df = pd.read_csv('output1.csv')

# Combine date and time into a single datetime column
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d-%m-%Y %I:%M %p')

# Select relevant features for clustering
features = ['datetime']

# Standardize the features (consider whether it's necessary)
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Specify the branching factor and threshold for BIRCH
branching_factor = 50  # Adjust this based on your data
threshold = 0.5  # Adjust this based on your data

# Perform BIRCH clustering
birch = Birch(branching_factor=branching_factor, threshold=threshold, n_clusters=None)
df['cluster'] = birch.fit_predict(df[features])

# Calculate cluster centroids
cluster_centroids = df.groupby('cluster')[features].mean()

# Calculate the centroid of centroids
centroid_of_centroids = cluster_centroids.mean()

# Visualize the clusters with a scatter plot
plt.scatter(df['datetime'], df['cluster'], c=df['cluster'], cmap='viridis', marker='o', s=50)
plt.scatter(cluster_centroids['datetime'], cluster_centroids.index, c='red', marker='x', s=100, label='Centroids')
plt.scatter(centroid_of_centroids['datetime'], len(cluster_centroids), c='blue', marker='o', s=150, label='Centroid of Centroids')
plt.xlabel('Datetime')
plt.ylabel('Cluster')
plt.title('Clusters based on Datetime (BIRCH) with Centroids')
plt.legend()
plt.show()
