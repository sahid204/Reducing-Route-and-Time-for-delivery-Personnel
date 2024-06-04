import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from hyperopt import fmin, tpe, hp

# Set a random seed for reproducibility
np.random.seed(42)
# Load data from CSV file
file_path = 'out_1.csv'  # Replace 'output1.csv' with the actual file path
df = pd.read_csv(file_path)

# Extract latitude and longitude columns
data = df[['latitude', 'longitude']].values

# Standardize the data
data = StandardScaler().fit_transform(data)

# Define an extended search space for hyperopt
space = {
    'eps': hp.uniform('eps', 0.1, 10.0),
    'min_samples': hp.choice('min_samples', [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30])
}

# Objective function to minimize (negative silhouette score)
def objective(params):
    dbscan = DBSCAN(eps=params['eps'], min_samples=int(params['min_samples']))
    labels_pred = dbscan.fit_predict(data)
    
    # Check if there's more than one unique label
    unique_labels = np.unique(labels_pred)
    if len(unique_labels) <= 1:
        return {'loss': -float('inf'), 'status': 'fail'}  # Return a large negative value for invalid cases
    
    # Compute silhouette score for valid cases
    score = -silhouette_score(data, labels_pred)
    
    return {'loss': score, 'status': 'ok', 'labels': labels_pred.tolist(), 'params': params}

# Run hyperparameter optimization
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50)

# Print the best parameters
print("Best Parameters:", best_params)

# Fit DBSCAN with the best parameters
best_dbscan = DBSCAN(eps=best_params['eps'], min_samples=int(best_params['min_samples']))
best_labels = best_dbscan.fit_predict(data)

# Print the number of unique clusters and silhouette score
unique_clusters = np.unique(best_labels)
print("Number of Unique Clusters:", len(unique_clusters))
print("Silhouette Score:", silhouette_score(data, best_labels))

# Plot each cluster separately with unique colors
for cluster_label in unique_clusters:
    cluster_points = data[best_labels == cluster_label]
    
    plt.scatter(
        cluster_points[:, 1],
        cluster_points[:, 0],
        label=f'Cluster {cluster_label}',
        s=50,
        edgecolors='k',
    )

plt.title('DBSCAN Clustering with Tuned Parameters')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()
