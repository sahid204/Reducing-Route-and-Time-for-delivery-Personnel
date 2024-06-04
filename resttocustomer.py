import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance

# Read customer and restaurant cluster data from CSV files
customer_clusters = pd.read_csv('output1.csv')
restaurant_clusters = pd.read_csv('restaurantoutput1.csv')

# Create a function to calculate distance between two points
def calculate_distance(lat1, lon1, lat2, lon2):
    return distance.euclidean((lat1, lon1), (lat2, lon2))

# Loop through each customer cluster
for index_c, row_c in customer_clusters.iterrows():
    min_distance = float('inf')
    nearest_cluster_id = None

    # Loop through each restaurant cluster
    for index_r, row_r in restaurant_clusters.iterrows():
        # Calculate distance between centroids
        dist = calculate_distance(row_c['latitude'], row_c['longitude'], row_r['Latitude'], row_r['Longitude'])

        # Update nearest cluster if current distance is smaller
        if dist < min_distance:
            min_distance = dist
            nearest_cluster_id = row_r['Unnamed: 0']

    # Assign the nearest restaurant cluster to the customer cluster
    customer_clusters.at[index_c, 'nearest_restaurant_cluster'] = nearest_cluster_id

    # Print information about each customer cluster and its nearest restaurant cluster
    print(f"Customer Cluster {row_c['Unnamed: 0']} is assigned to Nearest Restaurant Cluster {nearest_cluster_id}")

# Save the result to a new CSV file
customer_clusters.to_csv('customer_clusters_with_nearest_restaurant.csv', index=False)

# Visualize customer and restaurant clusters
plt.figure(figsize=(10, 6))

# Plot customer clusters
sns.scatterplot(x='longitude', y='latitude', hue='Unnamed: 0', data=customer_clusters, palette='Set1', s=100, label='Customer Clusters')

# Plot restaurant clusters
sns.scatterplot(x='Longitude', y='Latitude', hue='Unnamed: 0', data=restaurant_clusters, palette='Set2', s=150, marker='s', label='Restaurant Clusters')

# Highlight the nearest restaurant cluster for each customer cluster
for index, row in customer_clusters.iterrows():
    nearest_cluster = row['nearest_restaurant_cluster']
    nearest_restaurant = restaurant_clusters[restaurant_clusters['Unnamed: 0'] == nearest_cluster]
    plt.plot(nearest_restaurant['Longitude'], nearest_restaurant['Latitude'], 'kx', markersize=12)

plt.title('Customer and Restaurant Clusters with Nearest Restaurant')
plt.legend()
plt.show()
