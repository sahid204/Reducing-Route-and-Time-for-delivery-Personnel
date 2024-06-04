import pandas as pd
import numpy as np
import math

# Load latitude and longitude data from a CSV file
df = pd.read_csv('out_2.csv')

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

# Iterate through each city as the starting point
for start_city in range(n):
    print(f"Starting Point (City Index): {start_city}")
    
    total_distance = 0.0
    tour = [start_city]
    
    unvisited_cities = set(range(n))
    unvisited_cities.remove(start_city)

    while unvisited_cities:
        min_insertion_cost = float('inf')
        best_insertion_position = None
        new_city = None

        for city in unvisited_cities:
            for i in range(len(tour)):
                current_cost = (
                    distance_matrix[tour[i]][city] + distance_matrix[city][tour[(i + 1) % len(tour)]] - distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
                )
                if current_cost < min_insertion_cost:
                    min_insertion_cost = current_cost
                    best_insertion_position = i
                    new_city = city

        if new_city is not None:
            tour.insert((best_insertion_position + 1) % len(tour), new_city)
            unvisited_cities.remove(new_city)
            total_distance += min_insertion_cost
        else:
            # If no suitable insertion found, break the loop
            break

    # Return to the starting city
    total_distance += distance_matrix[tour[-1]][start_city]
    tour.append(start_city)

    print(f"Optimal Tour (City Indices): {tour}")
    print(f"Total Distance (in kilometers): {total_distance}")
    print()
