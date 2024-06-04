import pandas as pd
from scipy.spatial import cKDTree
import numpy as np

# Load customer and restaurant data
customer_data = pd.read_csv("cust_time.csv")
restaurant_data = pd.read_csv("restaurantoutputtime.csv")

# Convert latitude and longitude to radians
customer_data['lat_rad'] = np.radians(customer_data['Latitude'])
customer_data['lon_rad'] = np.radians(customer_data['Longitude'])
restaurant_data['lat_rad'] = np.radians(restaurant_data['Latitude'])
restaurant_data['lon_rad'] = np.radians(restaurant_data['Longitude'])

# Build a KD-tree for restaurants
tree_restaurant = cKDTree(restaurant_data[['lat_rad', 'lon_rad']])

# Function to find nearest restaurant for each customer
# Function to find nearest restaurant for each customer
def find_nearest_restaurant(row):
    dist, idx = tree_restaurant.query([row['lat_rad'], row['lon_rad']])
    nearest_restaurant = restaurant_data.iloc[idx]
    distance = dist * 6371  # Convert distance from radians to kilometers
    return pd.Series({
        'Customer Latitude': row['Latitude'],
        'Customer Longitude': row['Longitude'],
        'Customer Date': row['date_x'],
        'Customer Time': row['time_x'],
        'Nearest Restaurant Latitude': nearest_restaurant['Latitude'],
        'Nearest Restaurant Longitude': nearest_restaurant['Longitude'],
        'Nearest Restaurant Date': nearest_restaurant['date_y'],  # Include restaurant date
        'Nearest Restaurant Time': nearest_restaurant['time_y'],  # Include restaurant time
        'Distance': distance
    })

# Filter customer data for a specific date
filtered_customer_data = customer_data[customer_data['date_x'] == '20-01-2024']

# Reset the index of filtered_customer_data
filtered_customer_data = filtered_customer_data.reset_index(drop=True)

# Apply the function to each customer
nearest_restaurant_info = filtered_customer_data.apply(find_nearest_restaurant, axis=1)

# Concatenate the result back to filtered_customer_data
filtered_customer_data = pd.concat([filtered_customer_data, nearest_restaurant_info], axis=1)

# Output the result to a CSV file
filtered_customer_data.to_csv('nearest_restaurant_info.csv', index=False)
