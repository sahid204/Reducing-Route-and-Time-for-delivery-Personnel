import pandas as pd
import math

# Load restaurant and customer data from CSV files
def load_data(restaurant_file, customer_file):
    restaurants_df = pd.read_csv(restaurant_file)
    customers_df = pd.read_csv(customer_file)
    return restaurants_df, customers_df

# Function to calculate distance between two locations
def calculate_distance(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Calculate the change in coordinates
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Calculate the distance
    distance = R * c
    return distance

# Function to find the nearest restaurant(s) for a customer
def find_nearest_restaurant(customer_lat, customer_lon, customer_time, customer_date, restaurants_df):
    nearest_restaurants = []
    min_distance = float('inf')
    restaurant_date_time = None

    for index, row in restaurants_df.iterrows():
        restaurant_lat, restaurant_lon = row['Latitude'], row['Longitude']
        distance = calculate_distance(customer_lat, customer_lon, restaurant_lat, restaurant_lon)
        # Convert 'time_y' and 'date_y' to datetime objects
        restaurant_time = pd.to_datetime(row['time_y']).time()
        restaurant_date = pd.to_datetime(row['date_y'], format='%d-%m-%Y').date()
        # Check if the restaurant can deliver within 30 minutes and on the specified date
        if distance < min_distance and restaurant_time >= customer_time and restaurant_date == customer_date:
            min_distance = distance
            nearest_restaurants = [index]
            restaurant_date_time = (restaurant_date, restaurant_time)  # Store restaurant date and time
        elif distance == min_distance and restaurant_time >= customer_time and restaurant_date == customer_date:
            nearest_restaurants.append(index)

    return nearest_restaurants, restaurant_date_time

# Function to assign customers to nearest restaurant(s) for a specific date
def assign_customers_for_date(restaurants_df, customers_df, target_date):
    assignments = []
    for index, customer_row in customers_df.iterrows():
        customer_lat, customer_lon = customer_row['lat1'], customer_row['long1']
        customer_time = pd.to_datetime(customer_row['time_x']).time()
        customer_date = pd.to_datetime(customer_row['date_x'], format='%d-%m-%Y').date()
        if customer_date != target_date:
            continue  # Skip customers with dates other than the target date
        nearest_restaurants, restaurant_date_time = find_nearest_restaurant(customer_lat, customer_lon, customer_time, target_date, restaurants_df)
        if nearest_restaurants:
            for restaurant_index in nearest_restaurants:
                # Get restaurant's latitude and longitude
                restaurant_lat = restaurants_df.loc[restaurant_index, 'Latitude']
                restaurant_lon = restaurants_df.loc[restaurant_index, 'Longitude']
                assignments.append({
                    'Customer': index, 
                    'Customer_Latitude': customer_lat, 
                    'Customer_Longitude': customer_lon,
                    'Restaurant': restaurant_index, 
                    'Restaurant_Latitude': restaurant_lat, 
                    'Restaurant_Longitude': restaurant_lon, 
                    'Customer_Date': customer_date, 
                    'Customer_Time': customer_time, 
                    'Restaurant_Date': restaurant_date_time[0], 
                    'Restaurant_Time': restaurant_date_time[1]
                })
        else:
            assignments.append({
                'Customer': index, 
                'Customer_Latitude': customer_lat, 
                'Customer_Longitude': customer_lon,
                'Restaurant': 'No available restaurant within 30 minutes', 
                'Customer_Date': customer_date, 
                'Customer_Time': customer_time
            })
    return pd.DataFrame(assignments)

# Output assignments
def output_assignments(assignments_df, output_file):
    assignments_df.to_csv(output_file, index=False)
    print("Assignments saved to", output_file)
 
# Load data
restaurants_df, customers_df = load_data('restaurantoutputtime.csv', 'cust_time.csv')

# Assign customers to restaurants for a specific date
target_date = pd.to_datetime('2024-01-19').date()  # Specify target date
assignments_df = assign_customers_for_date(restaurants_df, customers_df, target_date)

# Output assignments to CSV file
output_file = 'assignments.csv'
output_assignments(assignments_df, output_file)
