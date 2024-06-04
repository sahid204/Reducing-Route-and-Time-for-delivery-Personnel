import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math

# Load restaurant and customer data from CSV
restaurants_df = pd.read_csv('restaurantoutputtime.csv')
customers_df = pd.read_csv('cust_time.csv')

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) * math.sin(dlat / 2)
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) * math.sin(dlon / 2)
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

# Function to create distance matrix between all locations
def create_distance_matrix(restaurants, customers):
    num_locations = len(restaurants) + len(customers)
    distance_matrix = [[0] * num_locations for _ in range(num_locations)]
    
    for i, (r_lat, r_lon) in enumerate(zip(restaurants['Latitude'], restaurants['Longitude'])):
        for j, (c_lat, c_lon) in enumerate(zip(customers['Latitude'], customers['Longitude'])):
            distance_matrix[i][len(restaurants) + j] = calculate_distance(r_lat, r_lon, c_lat, c_lon)
            distance_matrix[len(restaurants) + j][i] = distance_matrix[i][len(restaurants) + j]  # symmetric
            
    return distance_matrix

# Function to solve the VRP using OR-Tools
def solve_vrp(distance_matrix):
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)

    return solution, manager, routing

# Function to print the solution
def print_solution(solution, manager, routing, restaurants, customers):
    if solution:
        index = routing.Start(0)
        route = ''
        while not routing.IsEnd(index):
            route += ' -> ' + str(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        print(route + ' -> ' + str(manager.IndexToNode(index)))

# Main function
def main():
    distance_matrix = create_distance_matrix(restaurants_df, customers_df)
    solution, manager, routing = solve_vrp(distance_matrix)
    print_solution(solution, manager, routing, restaurants_df, customers_df)

if __name__ == '__main__':
    main()
