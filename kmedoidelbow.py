import pandas as pd
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('out_1.csv')

# Assuming your CSV file has columns named 'latitude' and 'longitude'
coordinates = data[['latitude', 'longitude']]

# Number of clusters to try (you can adjust this range based on your needs)
k_values = range(2, 11)  # Start from 2 clusters, as silhouette score requires at least 2 clusters

# Initialize lists to store results
smd = []
silhouette_scores = []

# Run K-medoids for each value of K and calculate SMD and silhouette score
for k in k_values:
    kmedoids = KMedoids(n_clusters=k, random_state=42, metric='manhattan')
    kmedoids.fit(coordinates)
    
    # Calculate SMD (using manhattan distance)
    smd.append(np.sum(np.min(kmedoids.transform(coordinates), axis=1)))
    
    # Calculate silhouette score
    silhouette_scores.append(silhouette_score(coordinates, kmedoids.labels_, metric='manhattan'))

# Plot the elbow curve
plt.figure(figsize=(12, 4))

# Plot SMD
plt.subplot(1, 2, 1)
plt.plot(k_values, smd, marker='o')
plt.title('Elbow Method for Optimal K (K-medoids)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Medoid Distances (SMD)')

# Plot silhouette score
plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score for Optimal K (K-medoids)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

# Find the optimal number of clusters (K) based on the elbow
optimal_k_elbow = smd.index(min(smd)) + 2  # Add 2 to start from K=2
print(f'Optimal number of clusters (K) based on elbow method: {optimal_k_elbow}')

# Find the optimal number of clusters (K) based on silhouette score
optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2  # Add 2 to start from K=2
print(f'Optimal number of clusters (K) based on silhouette score: {optimal_k_silhouette}')

# Choose the optimal number of clusters based on your preferred method (elbow or silhouette)
optimal_k = optimal_k_elbow

# Run K-medoids with the optimal number of clusters
optimal_kmedoids = KMedoids(n_clusters=optimal_k, random_state=42, metric='manhattan')
optimal_kmedoids.fit(coordinates)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(data['longitude'], data['latitude'], c=optimal_kmedoids.labels_, cmap='viridis', alpha=0.8)
plt.title(f'K-medoids Clustering with {optimal_k} Clusters')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
