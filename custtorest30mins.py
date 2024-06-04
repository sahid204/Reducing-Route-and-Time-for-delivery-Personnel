import pandas as pd
import geopy.distance
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
delivery_person_location = (12.9698, 77.75)  # Starting location of the delivery person
unvisited_customers = customers_df[['lat1', 'long1']].apply(tuple, axis=1).tolist()
route = [delivery_person_location]  # Start with the delivery person's location

# Nearest neighbor algorithm
while unvisited_customers:
    min_distance = float('inf')
    nearest_restaurant = None
    
    # Find nearest restaurant from current location
    for idx, restaurant in restaurants_df.iterrows():
        distance = calculate_distance(delivery_person_location, get_coords(restaurant))
        if distance < min_distance:
            min_distance = distance
            nearest_restaurant = get_coords(restaurant)
    
    # Update delivery person's location to the restaurant
    delivery_person_location = nearest_restaurant
    route.append(nearest_restaurant)
    
    # Find nearest unvisited customer from the restaurant
    min_distance = float('inf')
    nearest_customer = None
    for customer_coords in unvisited_customers:
        distance = calculate_distance(delivery_person_location, customer_coords)
        if distance < min_distance:
            min_distance = distance
            nearest_customer = customer_coords
    
    # Update delivery person's location to the customer and remove it from unvisited customers
    delivery_person_location = nearest_customer
    route.append(nearest_customer)
    unvisited_customers.remove(nearest_customer)

# Initialize plot
fig, ax = plt.subplots()

# Plot restaurants
for _, restaurant in restaurants_df.iterrows():
    ax.plot(restaurant['Latitude'], restaurant['Longitude'], 'red', label='Restaurant')

# Plot customers
for customer_coords in unvisited_customers:
    ax.plot(customer_coords[0], customer_coords[1], 'blue', label='Customer')

# Function to update plot for each frame
def update(frame):
    ax.clear()
    ax.set_title('Delivery Route')
    ax.plot(*zip(*route[:frame]), marker='o', linestyle='-', color='green')
    ax.annotate('Delivery Person', xy=route[frame], xytext=(route[frame][0], route[frame][1]+0.01), arrowprops=dict(facecolor='black', arrowstyle='->'))
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    return ax

# Create animation
ani = FuncAnimation(fig, update, frames=len(route), interval=1000)

# Display animation
plt.show()
