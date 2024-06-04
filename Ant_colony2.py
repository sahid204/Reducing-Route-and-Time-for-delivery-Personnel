import pandas as pd
import numpy as np
import random
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

# ACO parameters
num_ants = 20
num_iterations = 100
alpha = 1.0  # Pheromone importance
beta = 2.0   # Distance importance
evaporation_rate = 0.5
pheromone_deposit = 100.0

# Initialize pheromone levels
pheromones = np.ones((n, n))

# Helper function to select the next city for an ant
def select_next_city(ant, pheromone, alpha, beta):
    current_city = ant[-1]
    unvisited_cities = [city for city in range(n) if city not in ant]

    probabilities = []
    total_prob = 0.0

    for city in unvisited_cities:
        distance_factor = 1.0 / ((distance_matrix[current_city][city] + 1e-6) ** beta)  # Add a small constant to avoid division by zero
        pheromone_factor = pheromone[current_city][city] ** alpha
        probability = pheromone_factor * distance_factor
        probabilities.append(probability)
        total_prob += probability

    # Normalize probabilities
    probabilities = [prob / total_prob for prob in probabilities]
    
    if total_prob == 0:
        # If total_prob is zero, assign equal probabilities to all unvisited cities
        equal_probability = 1.0 / len(unvisited_cities)
        probabilities = [equal_probability] * len(unvisited_cities)

    # Choose the next city based on probabilities
    next_city = random.choices(unvisited_cities, probabilities)[0]
    return next_city

# ACO main loop
best_tour = None
best_distance = float('inf')

for iteration in range(num_iterations):
    ant_tours = []

    for ant in range(num_ants):
        current_tour = [random.randint(0, n - 1)]  # Start from a random city

        while len(current_tour) < n:
            next_city = select_next_city(current_tour, pheromones, alpha, beta)
            current_tour.append(next_city)

        # Complete the tour by returning to the starting city
        current_tour.append(current_tour[0])
        ant_tours.append(current_tour)

    # Update pheromone levels
    pheromones *= (1 - evaporation_rate)  # Evaporation
    for tour in ant_tours:
        tour_distance = sum(distance_matrix[tour[i]][tour[i + 1]] for i in range(n))
        pheromone_deposit_value = pheromone_deposit / tour_distance
        for i in range(n):
            pheromones[tour[i]][tour[i + 1]] += pheromone_deposit_value

    # Find the best tour of this iteration
    current_best_tour = min(ant_tours, key=lambda tour: sum(distance_matrix[tour[i]][tour[i + 1]] for i in range(n)))
    current_best_distance = sum(distance_matrix[current_best_tour[i]][current_best_tour[i + 1]] for i in range(n))

    # Update global best tour
    if current_best_distance < best_distance:
        best_tour = current_best_tour
        best_distance = current_best_distance

# Print the best tour and its total distance
print("Best Tour (City Indices):", best_tour)
print("Total Distance (in kilometers):", best_distance) 