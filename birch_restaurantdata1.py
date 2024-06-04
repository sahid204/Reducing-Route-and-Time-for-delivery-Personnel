import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import Birch

# Read data from Excel
data_path = 'restaurantoutput1.csv'   # Replace with the actual path to your Excel file
data = pd.read_csv(data_path)

# Assuming your data is in a DataFrame named 'data'

# Preprocess the data if necessary

# Specify the number of clusters you want
n_clusters = 3 #optimal value is 5

# Create a Birch instance
birch = Birch(n_clusters=n_clusters)

# Fit the algorithm to your data
birch.fit(data)

# Get the cluster labels for each data point
cluster_labels = birch.predict(data)

# Add the cluster labels to your DataFrame
data['cluster'] = cluster_labels

# Visualize the clusters
plt.scatter(data['Latitude'], data['Longitude'], c=data['cluster'], cmap='viridis')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Birch Clustering Results')
plt.colorbar(label='Cluster')
plt.show()