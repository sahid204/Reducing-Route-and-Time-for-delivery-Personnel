import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from hyperopt import fmin, tpe, hp

# Load data from CSV file
file_path = 'out_1.csv'  # Replace 'output1.csv' with the actual file path
df = pd.read_csv(file_path)

# Extract latitude and longitude columns
data = df[['latitude', 'longitude']].values

# Standardize the data
data = StandardScaler().fit_transform(data)

# Manually try different values for eps and min_samples
eps_values = [0.5, 1.0, 1.5, 2.0]
min_samples_values = [3, 5, 10, 15]

best_silhouette = -1
best_eps = None
best_min_samples = None
best_labels = None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data)

        # Exclude noise points (-1) from silhouette calculation
        if len(set(labels)) > 1:
            silhouette = silhouette_score(data, labels)
            print(f'eps={eps}, min_samples={min_samples}, Silhouette Score: {silhouette}')

            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_eps = eps
                best_min_samples = min_samples
                best_labels = labels

# Visualize the best clustering
plt.scatter(data[:, 0], data[:, 1], c=best_labels, cmap='viridis', marker='o')
plt.title('Best DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

print(f'Best Parameters: eps={best_eps}, min_samples={best_min_samples}')
print(f'Number of Unique Clusters: {len(set(best_labels))}')
print(f'Silhouette Score: {best_silhouette}')
