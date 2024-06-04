import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# Load customer and restaurant data from CSV files
customer_data = pd.read_csv("cust_time.csv")
restaurant_data = pd.read_csv("restaurantoutputtime.csv")

# Convert latitude and longitude to radians
customer_data['lat_rad'] = np.radians(customer_data['Latitude'])
customer_data['lon_rad'] = np.radians(customer_data['Longitude'])
restaurant_data['lat_rad'] = np.radians(restaurant_data['Latitude'])
restaurant_data['lon_rad'] = np.radians(restaurant_data['Longitude'])

# Sort restaurant data based on date and time
restaurant_data = restaurant_data.sort_values(by=['date_y', 'time_y']).reset_index(drop=True)

# Build a KD-Tree for restaurant locations
tree_restaurant = cKDTree(restaurant_data[['lat_rad', 'lon_rad']])

# Function to find nearest restaurant for each customer
def find_nearest_restaurant(row):
    dist, idx = tree_restaurant.query([row['lat_rad'], row['lon_rad']])
    nearest_restaurant = restaurant_data.iloc[idx]
    # Filter restaurants based on date and time
    nearest_restaurant = nearest_restaurant[
        (nearest_restaurant['date_y'] == row['date_x']) & 
        (nearest_restaurant['time_y'] == row['time_x'])
    ]
    return nearest_restaurant  # Return the nearest restaurant(s) information

# Apply the function to each customer
customer_restaurant_mapping = customer_data.apply(find_nearest_restaurant, axis=1)

# Output the result to a CSV file
customer_restaurant_mapping.to_csv('customer_restaurant_mapping.csv', index=False)
