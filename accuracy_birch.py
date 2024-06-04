import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score

# Read data from Excel
data_path = 'out_3.csv'   # Replace with the actual path to your Excel file
data = pd.read_csv(data_path)

# Preprocess the data if necessary

# Specify the number of clusters you want
n_clusters = 3

# Create a Birch instance
birch = Birch(n_clusters=n_clusters)

# Fit the algorithm to your data
birch.fit(data)

# Get the cluster labels for each data point
cluster_labels = birch.predict(data)

# Add the cluster labels to your DataFrame
data['cluster'] = cluster_labels

# Calculate the silhouette score
silhouette_avg = silhouette_score(data[['latitude', 'longitude']], cluster_labels)

# Normalize silhouette score to percentage
silhouette_percentage = (silhouette_avg + 1) * 50 

print(f"Silhouette Score (Percentage): {silhouette_percentage:.2f}%")

# Visualize the clusters
plt.scatter(data['latitude'], data['longitude'], c=data['cluster'], cmap='viridis')
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.title('Birch Clustering Results')
plt.colorbar(label='Cluster')
plt.show()
