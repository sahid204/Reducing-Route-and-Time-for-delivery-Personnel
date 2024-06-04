import pandas as pd
from sklearn.cluster import KMeans
from geopy.distance import geodesic
import numpy as np
import matplotlib.pyplot as plt

# Load data
customer_data = pd.read_csv('cust_time.csv')
delivery_boys_locations = [(12.95404312,77.56886512), (12.91804385,77.63768893), (12.91192952, 77.63801917)]
restaurant_location = (12.96396971,77.63859853)  # Restaurant location

# Calculate distance between points
def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).kilometers

# Cluster customers
customer_coordinates = customer_data[['lat1', 'long1']].values
kmeans = KMeans(n_clusters=3, random_state=0).fit(customer_coordinates)
customer_data['cluster'] = kmeans.labels_

# Find nearest delivery boy to restaurant
nearest_delivery_boy_index = min(range(len(delivery_boys_locations)), key=lambda i: calculate_distance(restaurant_location, delivery_boys_locations[i]))

# Calculate distance each delivery boy travels to reach restaurant
distance_to_restaurant = [calculate_distance(restaurant_location, loc) for loc in delivery_boys_locations]

# Assign each delivery boy to the nearest available cluster
assigned_clusters = []
assigned_centers = []  # List to keep track of assigned cluster centers
for loc in delivery_boys_locations:
    min_distance = float('inf')
    nearest_cluster = None
    for i, cluster_center in enumerate(kmeans.cluster_centers_):
        if i not in assigned_centers:  # Check if cluster center is available
            distance = calculate_distance(cluster_center, loc)
            if distance < min_distance:
                min_distance = distance
                nearest_cluster = i
    assigned_clusters.append(nearest_cluster)
    assigned_centers.append(nearest_cluster)

# Print distance each delivery boy travels to reach restaurant
for i, distance in enumerate(distance_to_restaurant):
    print(f"Delivery Boy {i+1} traveled {distance:.2f} kilometers to reach the restaurant.")

# Print cluster each delivery boy is assigned to
for i, cluster_index in enumerate(assigned_clusters):
    print(f"Delivery Boy {i+1} is assigned to Cluster {cluster_index+1}.")
plt.figure(figsize=(10, 8))
plt.scatter(customer_data['lat1'], customer_data['long1'], c=customer_data['cluster'], cmap='viridis', label='Customer Clusters', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=100, c='black', label='Cluster Centers')
plt.scatter(delivery_boys_locations[0][0], delivery_boys_locations[0][1], c='red', label='Delivery Boy 1')
plt.scatter(delivery_boys_locations[1][0], delivery_boys_locations[1][1], c='blue', label='Delivery Boy 2')
plt.scatter(delivery_boys_locations[2][0], delivery_boys_locations[2][1], c='green', label='Delivery Boy 3')
plt.scatter(restaurant_location[0], restaurant_location[1], c='orange', label='Restaurant')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Assignment of Delivery Boys to Restaurant and Customer Clusters')
plt.legend()
plt.grid(True)
plt.show()