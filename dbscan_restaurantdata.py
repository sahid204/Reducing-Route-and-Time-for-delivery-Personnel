import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Load data from Excel
data = pd.read_csv("Bangaloredata.csv")  # Replace with your Excel file path

# Extract the columns you want to use for clustering

X = data[["Latitude", "Longitude"]].values

# Create a DBSCAN model
dbscan = DBSCAN(eps=2.4751, min_samples=4)  # Adjust parameters as needed

# Fit the model and get cluster labels
cluster_labels = dbscan.fit_predict(X)

# Separate noise points from clusters
noise_mask = cluster_labels == -1
clusters_mask = ~noise_mask

# Plot the clusters in one color and noise in another color
plt.figure(figsize=(10, 6))
plt.scatter(X[clusters_mask, 0], X[clusters_mask, 1], c=cluster_labels[clusters_mask], cmap="viridis", label="Clusters")
plt.scatter(X[noise_mask, 0], X[noise_mask, 1], c='red', marker='x', label="Noise")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.title("DBSCAN Clustering with Noise")
plt.legend()
plt.colorbar(label="Cluster Label")
plt.show()