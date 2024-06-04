import random
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import matplotlib.pyplot as plt

# Function to calculate the distance between two points given their latitude and longitude
def calculate_distance(lat1, lon1, lat2, lon2):
    coord1 = (lat1, lon1)
    coord2 = (lat2, lon2)
    return geodesic(coord1, coord2).kilometers

def nearest_neighbor_path(latitude, longitude):
    num_points = len(latitude)
    unvisited = list(range(num_points))
    start_point = random.choice(unvisited)
    current_point = start_point
    path = [current_point]
    unvisited.remove(current_point)

    while unvisited:
        nearest_point = min(unvisited, key=lambda point: calculate_distance(latitude[current_point], longitude[current_point], latitude[point], longitude[point]))
        current_point = nearest_point
        path.append(current_point)
        unvisited.remove(current_point)

    path.append(start_point)  # Return to the starting point to complete the loop
    return path

# Example usage
data_file = 'out_3.csv'  # Replace with the path to your CSV file containing latitude and longitude data

# Load latitude and longitude data from a CSV file
df = pd.read_csv(data_file)
latitude = df['latitude'].values
longitude = df['longitude'].values

tsp_path_indices = nearest_neighbor_path(latitude, longitude)

# Create latitude and longitude arrays for the TSP path
tsp_path_latitudes = [latitude[i] for i in tsp_path_indices]
tsp_path_longitudes = [longitude[i] for i in tsp_path_indices]

# Calculate the total distance of the TSP path
total_distance = sum(calculate_distance(latitude[tsp_path_indices[i]], longitude[tsp_path_indices[i]],
                                        latitude[tsp_path_indices[i + 1]], longitude[tsp_path_indices[i + 1]])
                     for i in range(len(tsp_path_indices) - 1))

# Visualize the TSP path with starting point and arrows
plt.figure(figsize=(10, 6))
plt.scatter(longitude, latitude, c='red', marker='o', label='Locations')
plt.plot(tsp_path_longitudes, tsp_path_latitudes, linestyle='-', linewidth=2, markersize=5, label='TSP Path', color='blue')

# Mark the starting point
start_lat = latitude[tsp_path_indices[0]]
start_lon = longitude[tsp_path_indices[0]]
plt.plot(start_lon, start_lat, 'go', label='Starting Point')

# Add arrows to indicate the path direction
for i in range(len(tsp_path_indices) - 1):
    plt.annotate("", xy=(tsp_path_longitudes[i+1], tsp_path_latitudes[i+1]), xytext=(tsp_path_longitudes[i], tsp_path_latitudes[i]),
                 arrowprops=dict(arrowstyle="->", lw=1.5))

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.title('Traveling Salesman Problem (TSP) Path Using Nearest Neighbor Algorithm')
plt.grid(True)
plt.show()

print("TSP Path Order (Indices):", tsp_path_indices)
print("Total Distance (in kilometers):", total_distance)
