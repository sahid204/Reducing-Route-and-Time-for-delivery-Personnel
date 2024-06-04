import random
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import time

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

# Measure the start time
start_time = time.time()

tsp_path_indices = nearest_neighbor_path(latitude, longitude)

# Measure the end time
end_time = time.time()

# Calculate the total distance of the TSP path
total_distance = sum(calculate_distance(latitude[tsp_path_indices[i]], longitude[tsp_path_indices[i]],
                                        latitude[tsp_path_indices[i + 1]], longitude[tsp_path_indices[i + 1]])
                     for i in range(len(tsp_path_indices) - 1))

# Visualize the TSP path with starting point and arrows (same as before)

# Print the TSP path and total distance
print("TSP Path Order (Indices):", tsp_path_indices)
print("Total Distance (in kilometers):", total_distance)

# Calculate the overall runtime
overall_runtime = end_time - start_time
print("Overall Algorithm Runtime: {:.2f} seconds".format(overall_runtime))
