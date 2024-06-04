import pandas as pd
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('out_1.csv')  # Replace 'your_file.csv' with your actual file path

# Select features
X = data[['latitude', 'longitude']]

# Initialize variables
n_clusters_target = 3
threshold = 0.1
branching_factor = 50

# Keep adjusting the threshold until a sufficient number of clusters are found
while True:
    # Create Birch instance
    birch = Birch(n_clusters=None, threshold=threshold, branching_factor=branching_factor)

    # Fit the model
    birch.fit(X)

    # Actual number of clusters found by Birch
    actual_n_clusters = birch.subcluster_centers_.shape[0]

    # If the number of clusters is greater than or equal to the target, break the loop
    if actual_n_clusters >= n_clusters_target:
        break

    # Adjust the threshold
    threshold *= 0.9  # You can experiment with different adjustment factors

# Predict cluster labels
labels = birch.predict(X)

# Add cluster labels to the DataFrame
data['cluster'] = labels

# Compute silhouette score
silhouette_avg = silhouette_score(X, labels)
print(f"Number of clusters found by Birch: {actual_n_clusters}")
print(f"Silhouette Score: {silhouette_avg}")

# Visualize the clusters (scatter plot)
plt.scatter(data['longitude'], data['latitude'], c=data['cluster'], cmap='viridis', s=50)
plt.title('Birch Clustering of Latitude and Longitude Data\nSilhouette Score: {:.2f}'.format(silhouette_avg))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
