import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# Load latitude and longitude data from a CSV file
df = pd.read_csv('out_3.csv')

# Convert latitude and longitude columns to NumPy arrays
latitude = df['latitude'].values
longitude = df['longitude'].values

# Function to calculate Haversine distance between two coordinates
def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1) 
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance

# Combine latitude and longitude into a 2D array
locations = np.column_stack((latitude, longitude))

# Calculate the distance matrix using Haversine distance
n = len(locations)
distance_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        distance_matrix[i][j] = haversine(latitude[i], longitude[i], latitude[j], longitude[j])

# Solve the TSP using linear_sum_assignment (Hungarian algorithm)
row_ind, col_ind = linear_sum_assignment(distance_matrix)

# Reorder the locations based on the optimal order
optimal_order = col_ind

# Create the optimal path
optimal_path = locations[optimal_order]

# Calculate the total distance of the optimal path
total_distance = distance_matrix[row_ind, col_ind].sum()

# Visualize the optimal path
plt.figure(figsize=(10, 6))
plt.scatter(longitude, latitude, c='red', marker='o', label='Locations')

# Plot the optimal path as a line
plt.plot(optimal_path[:, 1], optimal_path[:, 0], linestyle='-', linewidth=2, markersize=5, label='Optimal Path', color='blue')

# Identify and mark the starting point
starting_point_lon, starting_point_lat = longitude[optimal_order[0]], latitude[optimal_order[0]]
plt.scatter(starting_point_lon, starting_point_lat, c='green', marker='x', s=100, label='Starting Point')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.title('Optimal Path with Starting Point')
plt.grid(True)
plt.show()

# Print the optimal path
print("Optimal Path Coordinates:")
for i, (lat, lon) in enumerate(optimal_path, start=1):
    print(f"Point {i}: Latitude: {lat}, Longitude: {lon}")

# Print the total distance
print(f"Total Distance: {total_distance} kilometers")
