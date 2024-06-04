import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load data from CSV file
file_path = 'restaurantoutput1.csv'  # Replace 'output1.csv' with the actual file path
df = pd.read_csv(file_path)

# Extract latitude and longitude columns
data = df[['Latitude', 'Longitude']].values

# Standardize the data
data = StandardScaler().fit_transform(data)

# Hyperparameters for hierarchical clustering
n_clusters = 3 # Number of clusters
linkage_type = 'ward'  # Linkage criterion: 'ward', 'complete', 'average', etc.

# Perform hierarchical clustering
hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
hierarchical_labels = hierarchical_cluster.fit_predict(data)

# Print the number of unique clusters
unique_clusters = np.unique(hierarchical_labels)
print("Number of Unique Clusters:", len(unique_clusters))

# Calculate and print the silhouette score
silhouette_avg = silhouette_score(data, hierarchical_labels)
print("Silhouette Score:", silhouette_avg)

# Plot each cluster separately with unique colors
for cluster_label in unique_clusters:
    cluster_points = data[hierarchical_labels == cluster_label]

    plt.scatter(
        cluster_points[:, 1],
        cluster_points[:, 0],
        label=f'Cluster {cluster_label}',
        s=50,
        edgecolors='k',
    )

plt.title('Hierarchical Clustering')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()
