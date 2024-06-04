import pandas as pd

# Load data from Excel
data = pd.read_csv("out_1.csv")

from sklearn_extra.cluster import KMedoids
import numpy as np

# Extract numeric columns
X = data.select_dtypes(include=[np.number])

# Create a KMedoids model
k = 3  # Number of clusters
kmedoids = KMedoids(n_clusters=k, random_state=0)

# Fit the model and obtain cluster labels and medoid indices
cluster_labels = kmedoids.fit_predict(X)
medoid_indices = kmedoids.medoid_indices_
import matplotlib.pyplot as plt

# Create scatter plots
for i in range(k):
    cluster_data = X.iloc[cluster_labels == i]
    medoid = X.iloc[medoid_indices[i]]
    plt.scatter(cluster_data.iloc[:,0], cluster_data.iloc[:,1], label=f'Cluster {i+1}')
    plt.scatter(medoid[0], medoid[1], marker='X', color='red', label=f'Medoid {i+1}')

plt.xlabel("latitude")
plt.ylabel("longitude")
plt.title("K-Medoids Clustering")
plt.legend()
plt.show()