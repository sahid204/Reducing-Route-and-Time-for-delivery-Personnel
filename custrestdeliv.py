import pandas as pd
import geopy.distance

# Load customer and restaurant data
customers_df = pd.read_csv("cust_time.csv")
restaurants_df = pd.read_csv("restaurantoutputtime.csv")

# Function to calculate distance between two points
def calculate_distance(coords_1, coords_2):
    return geopy.distance.geodesic(coords_1, coords_2).kilometers

# Function to get coordinates as a tuple
def get_coords(row):
    return (row['Latitude'], row['Longitude'])

# Initialize variables
current_location = (12.9698, 77.75)  # Starting location of the delivery person
unvisited_customers = customers_df[['lat1', 'long1']].apply(tuple, axis=1).tolist()
route = []

# Nearest neighbor algorithm
customer_counter = 1
restaurant_counter = 1
while unvisited_customers:
    min_distance = float('inf')
    nearest_restaurant = None
    
    # Find nearest restaurant from current location
    for idx, restaurant in restaurants_df.iterrows():
        distance = calculate_distance(current_location, get_coords(restaurant))
        if distance < min_distance:
            min_distance = distance
            nearest_restaurant = ("Restaurant " + str(restaurant_counter), get_coords(restaurant))  # Assign restaurant name
    
    # Update current location to the restaurant
    current_location = nearest_restaurant[1]
    route.append(nearest_restaurant)
    
    # Find nearest unvisited customer from the restaurant
    min_distance = float('inf')
    nearest_customer = None
    for customer_coords in unvisited_customers:
        distance = calculate_distance(current_location, customer_coords)
        if distance < min_distance:
            min_distance = distance
            nearest_customer = ("Customer " + str(customer_counter), customer_coords)  # Assign customer name
    
    # Update current location to the customer and remove it from unvisited customers
    current_location = nearest_customer[1]
    route.append(nearest_customer)
    unvisited_customers.remove(nearest_customer[1])
    
    # Increment counters for next customer and restaurant
    customer_counter += 1
    restaurant_counter += 1

# Add starting location to the end of the route
route.append(("Starting Location", (12.9698, 77.75)))

# Print the route with customer and restaurant names and coordinates
print("Optimized route:")
for location in route:
    print(location[0], location[1])
