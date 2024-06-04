import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from hyperopt import fmin, tpe, hp

# Set a random seed for reproducibility
np.random.seed(42)

# Load data from CSV file
file_path = 'restaurantoutput1.csv'  # Replace 'output1.csv' with the actual file path
df = pd.read_csv(file_path)

# Extract latitude and longitude columns
data = df[['Latitude', 'Longitude']].values

# Standardize the data
data = StandardScaler().fit_transform(data)

# Define an extended search space for hyperopt
space = {
    'eps': hp.uniform('eps', 0.1, 10.0),
    'min_samples': hp.choice('min_samples', [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30])
}

# Objective function to minimize (negative silhouette score)
def objective(params):
    try:
        dbscan = DBSCAN(eps=params['eps'], min_samples=int(params['min_samples']))
        labels_pred = dbscan.fit_predict(data)

        # Check if there's more than one unique label
        unique_labels = np.unique(labels_pred)
        if len(unique_labels) <= 1:
            return {'loss': -float('inf'), 'status': 'fail'}  # Return a large negative value for invalid cases

        # Compute silhouette score for valid cases
        score = -silhouette_score(data, labels_pred)
        return {'loss': score, 'status': 'ok', 'labels': labels_pred.tolist(), 'params': params}

    except Exception as e:
        print(f"Error in trial: {e}")
        return {'loss': -float('inf'), 'status': 'fail'}

# Run hyperparameter optimization
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50)

if 'eps' in best_params:
    best_dbscan = DBSCAN(eps=best_params['eps'] * 1.5, min_samples=int(best_params['min_samples']))
    best_labels = best_dbscan.fit_predict(data)

    # Apply hierarchical clustering as a post-processing step
    hierarchical_cluster = AgglomerativeClustering(n_clusters=5, linkage='ward')
    hierarchical_labels = hierarchical_cluster.fit_predict(data)

    # Plot the original DBSCAN clusters
    for cluster_label in np.unique(best_labels):
        cluster_points = data[best_labels == cluster_label]

        plt.scatter(
            cluster_points[:, 1],
            cluster_points[:, 0],
            label=f'DBSCAN Cluster {cluster_label}',
            s=50,
            marker='o',  # Use a different marker for DBSCAN
        )

    # Plot the refined clusters after hierarchical clustering
    for cluster_label in np.unique(hierarchical_labels):
        cluster_points = data[hierarchical_labels == cluster_label]

        plt.scatter(
            cluster_points[:, 1],
            cluster_points[:, 0],
            label=f'Refined Cluster {cluster_label}',
            s=50,
            marker='x',  # Use 'x' marker for refinement
        )

    plt.title('DBSCAN Clustering with Refinement')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.show()
else:
    print("Hyperparameter optimization failed. Check the code and dataset.")
