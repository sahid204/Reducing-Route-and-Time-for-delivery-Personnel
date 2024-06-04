import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('out_1.csv')

# Assuming your CSV file has columns named 'latitude' and 'longitude'
coordinates = data[['latitude', 'longitude']]

# Number of clusters to try (you can adjust this range based on your needs)
k_values = range(2, 11)  # Start from 2 clusters, as silhouette score requires at least 2 clusters

# Initialize lists to store results
sse = []
silhouette_scores = []

# Run K-means for each value of K and calculate SSE and silhouette score
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(coordinates)
    
    # Calculate SSE
    sse.append(kmeans.inertia_)
    
    # Calculate silhouette score
    silhouette_scores.append(silhouette_score(coordinates, kmeans.labels_))

# Plot the elbow curve
plt.figure(figsize=(12, 4))

# Plot SSE
plt.subplot(1, 2, 1)
plt.plot(k_values, sse, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Distances (SSE)')

# Plot silhouette score
plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

# Find the optimal number of clusters (K) based on the elbow
optimal_k_elbow = sse.index(min(sse)) + 2  # Add 2 to start from K=2
print(f'Optimal number of clusters (K) based on elbow method: {optimal_k_elbow}')

# Find the optimal number of clusters (K) based on silhouette score
optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2  # Add 2 to start from K=2
print(f'Optimal number of clusters (K) based on silhouette score: {optimal_k_silhouette}')

# Choose the optimal number of clusters based on your preferred method (elbow or silhouette)
optimal_k = optimal_k_elbow

# Run K-means with the optimal number of clusters
optimal_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
optimal_kmeans.fit(coordinates)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(data['longitude'], data['latitude'], c=optimal_kmeans.labels_, cmap='viridis', alpha=0.8)
plt.title(f'K-means Clustering with {optimal_k} Clusters')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
