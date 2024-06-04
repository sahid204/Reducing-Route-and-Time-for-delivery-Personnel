import pandas as pd
import numpy as np
import math

# Load latitude and longitude data from a CSV file
df = pd.read_csv('out_1.csv')

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

# Calculate the distance matrix using Haversine distance
n = len(latitude)
distance_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        distance_matrix[i][j] = haversine(latitude[i], longitude[i], latitude[j], longitude[j])

# Two-Opt Algorithm to find the optimal path
def two_opt(path, distance_matrix):
    n = len(path)
    improved = True

    while improved:
        improved = False

        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue  # No need to consider adjacent cities
                new_path = path[:]
                new_path[i:j] = path[j - 1:i - 1:-1]  # Reverse the path segment

                current_distance = sum(distance_matrix[path[k]][path[k + 1]] for k in range(n - 1))
                new_distance = sum(distance_matrix[new_path[k]][new_path[k + 1]] for k in range(n - 1))

                if new_distance < current_distance:
                    path = new_path
                    improved = True

    return path

# Iterate through each city as a starting point
for starting_point in range(n):
    # Create an initial path starting from the current city
    initial_path = list(range(n))
    initial_path.remove(starting_point)
    initial_path.insert(0, starting_point)

    # Improve the initial path using Two-Opt
    optimal_path = two_opt(initial_path, distance_matrix)

    # Add the starting point to the end to complete the loop
    optimal_path.append(optimal_path[0])

    # Calculate the total distance covered
    total_distance = sum(distance_matrix[optimal_path[i]][optimal_path[i + 1]] for i in range(n))

    # Print the optimal path and total distance for the current starting point
    print(f'Starting Point: {starting_point}')
    print(f'Optimal Path: {optimal_path}')
    print(f'Total Distance: {total_distance:.2f} km')
    print('-' * 50)  # Separating lines for clarity
