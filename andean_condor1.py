import random
import numpy as np
import pandas as pd
from geopy.distance import geodesic
 
# Function to calculate the distance between two points given their latitude and longitude
def calculate_distance(lat1, lon1, lat2, lon2):
    coord1 = (lat1, lon1)
    coord2 = (lat2, lon2)
    return geodesic(coord1, coord2).kilometers

def RandomNumber(low, high):
    # Simulate a random number between low (inclusive) and high (exclusive)
    return random.randint(low, high - 1)

def Validation(machine_part, machine_cell, part_cell, max_machines_per_cell, max_parts_per_cell):
    # Check the constraint for machines
    for machine_index, machine_row in enumerate(machine_cell):
        total_cells_assigned = sum(machine_row)
        if total_cells_assigned > max_machines_per_cell:
            return False  # Constraint violated

    # Check the constraint for parts
    for part_index, part_row in enumerate(part_cell):
        total_cells_assigned = sum(part_row)
        if total_cells_assigned > max_parts_per_cell:
            return False  # Constraint violated

    # If all constraints are met, return True
    return True

def Randomization(machine_cell, machines, cells):
    M = machines
    C = cells

    while True:
        # Randomly select a machine
        random_machine = RandomNumber(0, M)

        # Reset all cells of the selected machine to 0
        for k in range(C):
            machine_cell[random_machine][k] = 0

        # Randomly select a cell and set it to 1 for the selected machine
        random_cell = RandomNumber(0, C)
        machine_cell[random_machine][random_cell] = 1

        # No constraint checking in Randomization

        return machine_cell

def Exploration(machine_part, machines, parts, cells, max_machines_per_cell, max_parts_per_cell, data_file):
    M = machines
    P = parts
    C = cells

    # Load latitude and longitude data from a CSV file
    df = pd.read_csv(data_file)
    latitude = df['latitude'].values
    longitude = df['longitude'].values

    machine_cell = np.zeros((M, C), dtype=int)
    part_cell = np.zeros((P, C), dtype=int)

    # Initialize variables to store the path and distance
    path = []
    total_distance = 0.0

    while True:
        # Generate the Machine x Cell matrix using Randomization
        machine_cell = Randomization(machine_cell, M, C)

        # Create the Part x Cell matrix from the data of the Machine x Cell matrix
        for j in range(P):
            temp_part = np.zeros(M, dtype=int)
            cell_count = np.zeros(C, dtype=int)

            for k in range(C):
                for i in range(M):
                    temp_part[i] = machine_cell[i][k] * machine_part[i][j]

                cell_count[k] = sum(temp_part)

            max_index = np.argmax(cell_count)
            part_cell[j][max_index] = 1

        # Verify if the created solution does not break the constraints
        flag = Validation(machine_part, machine_cell, part_cell, max_machines_per_cell, max_parts_per_cell)

        if flag:
            break

    # Create a path by traversing machines in random order
    remaining_machines = list(range(M))
    current_machine = remaining_machines[0]
    remaining_machines.pop(0)
    path.append((latitude[current_machine], longitude[current_machine]))

    while remaining_machines:
        next_machine_index = RandomNumber(0, len(remaining_machines))
        next_machine = remaining_machines[next_machine_index]
        remaining_machines.pop(next_machine_index)

        next_lat, next_lon = latitude[next_machine], longitude[next_machine]
        total_distance += calculate_distance(latitude[current_machine], longitude[current_machine], next_lat, next_lon)
        path.append((next_lat, next_lon))
        current_machine = next_machine

    # Return to the starting point to complete the loop
    total_distance += calculate_distance(latitude[current_machine], longitude[current_machine], path[0][0], path[0][1])
    path.append((path[0][0], path[0][1]))

    return path, total_distance

# Example usage
machines = 5  # Replace with your desired number of machines
parts = 4     # Replace with your desired number of parts
cells = 3     # Replace with your desired number of cells
max_machines_per_cell = 2  # Maximum machines per cell constraint
max_parts_per_cell = 2     # Maximum parts per cell constraint
data_file = 'out_3.csv'  # Replace with the path to your CSV file containing latitude and longitude data

machine_part = np.random.randint(2, size=(machines, parts))  # Example random machine-part matrix

path, total_distance = Exploration(machine_part, machines, parts, cells, max_machines_per_cell, max_parts_per_cell, data_file)
print("Path:")
print(path)
print("Total Distance (in kilometers):", total_distance)
