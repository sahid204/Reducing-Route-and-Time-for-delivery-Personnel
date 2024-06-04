import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('out_1.csv')

# Assuming your CSV file has columns named 'latitude' and 'longitude'
coordinates = data[['latitude', 'longitude']]

# Number of clusters to try (you can adjust this range based on your needs)
k_values = range(1, 11)

# Initialize an empty list to store the sum of squared distances (SSE) for each K
sse = []

# Run K-means for each value of K and calculate SSE
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(coordinates)
    sse.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(k_values, sse, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Distances (SSE)')
plt.show()

# Find the optimal number of clusters (K) based on the elbow
optimal_k = sse.index(min(sse)) + 1
print(f'Optimal number of clusters (K): {optimal_k}')
