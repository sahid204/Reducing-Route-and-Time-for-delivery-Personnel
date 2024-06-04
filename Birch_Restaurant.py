import pandas as pd
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('restaurantoutput1.csv')  # Replace 'your_file.csv' with your actual file path

# Select features
X = data[['Latitude', 'Longitude']]

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Experiment with different hyperparameter values
birch = Birch(n_clusters=3, threshold=0.2, branching_factor=250)  # Adjust parameters accordingly

# Fit the model
birch.fit(X_scaled)

# Predict cluster labels
labels = birch.predict(X_scaled)

# Add cluster labels to the DataFrame
data['cluster'] = labels

# Compute silhouette score
silhouette_avg = silhouette_score(X_scaled, labels)
print(f"Number of clusters found by Birch: {birch.subcluster_centers_.shape[0]}")
print(f"Silhouette Score: {silhouette_avg}")

# Visualize the clusters (scatter plot)
plt.scatter(data['Longitude'], data['Latitude'], c=data['cluster'], cmap='viridis', s=50)
plt.title('Birch Clustering of Latitude and Longitude Data\nSilhouette Score: {:.2f}'.format(silhouette_avg))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
