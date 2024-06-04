import pandas as pd
from sklearn.cluster import KMeans
from geopy.distance import geodesic
import numpy as np

# Load data
customer_data = pd.read_csv('cust_time.csv')
delivery_boys_locations = [(12.95404312, 77.56886512), (12.91804385, 77.63768893), (12.91192952, 77.63801917), (12.93343554, 77.61434499), (12.91804385, 77.63768893), (12.93658526, 77.61346791)]
restaurant_location = (12.96396971, 77.63859853)  # Restaurant location

# Calculate distance between points
def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).kilometers

# Cluster customers
customer_coordinates = customer_data[['lat1', 'long1']].values
kmeans = KMeans(n_clusters=3, random_state=0).fit(customer_coordinates)
customer_data['cluster'] = kmeans.labels_

# Calculate distance each delivery boy travels to reach the restaurant
distance_to_restaurant = [calculate_distance(restaurant_location, loc) for loc in delivery_boys_locations]

# Assign each delivery boy to the nearest cluster
assigned_clusters = []
for loc in delivery_boys_locations:
    # Calculate distance to each cluster center
    distances_to_clusters = [calculate_distance(loc, cluster_center) for cluster_center in kmeans.cluster_centers_]
    # Assign to nearest cluster
    nearest_cluster = np.argmin(distances_to_clusters)
    assigned_clusters.append(nearest_cluster)

# Print distance each delivery boy travels to reach restaurant and the assigned cluster
for i, (distance, cluster_index) in enumerate(zip(distance_to_restaurant, assigned_clusters)):
    print(f"Delivery Boy {i+1} traveled {distance:.2f} kilometers to reach the restaurant and is assigned to Cluster {cluster_index+1}.")
