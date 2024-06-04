import pandas as pd
from scipy.spatial import distance

# Read customer and restaurant cluster data from CSV files
customer_clusters = pd.read_csv('output1.csv')
restaurant_clusters = pd.read_csv('restaurantoutput1.csv')

# Create a function to calculate distance between two points
def calculate_distance(lat1, lon1, lat2, lon2):
    return distance.euclidean((lat1, lon1), (lat2, lon2))

# Create an empty DataFrame to store the assignments
assignments = pd.DataFrame(columns=['customer_cluster_id', 'nearest_restaurant_cluster_id'])

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

    # Append the assignment to the new DataFrame
    assignments = pd.concat([assignments, pd.DataFrame({'customer_cluster_id': [row_c['Unnamed: 0']], 'nearest_restaurant_cluster_id': [nearest_cluster_id]})], ignore_index=True)

# Save the assignments to a new CSV file
assignments.to_csv('customer_to_restaurant_assignments.csv', index=False)
