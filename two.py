import pandas as pd

# Define the specific dates
specific_dates_restaurants = ['19-01-2024']  # Example specific dates for restaurants
specific_dates_customers = ['19-01-2024']  # Example specific dates for customers

# Load restaurant and customer data from CSV for specific dates
restaurants_df = pd.read_csv('restaurantoutputtime.csv', parse_dates=[['date_y', 'time_y']], dayfirst=True)
restaurants_df = restaurants_df[restaurants_df['date_y_time_y'].dt.strftime('%d-%m-%Y').isin(specific_dates_restaurants)]

customers_df = pd.read_csv('cust_time.csv', parse_dates=[['date_x', 'time_x']])
customers_df = customers_df[customers_df['date_x_time_x'].dt.strftime('%d-%m-%Y').isin(specific_dates_customers)]

# Define a function to calculate the time difference between two times
def calculate_time_difference(time1, time2):
    return abs((time2 - time1).total_seconds())

# Initialize a dictionary to store the nearest restaurant for each customer
nearest_restaurants = {}

# Iterate over each customer
for _, customer_row in customers_df.iterrows():
    customer_time = customer_row['date_x_time_x']
    nearest_restaurant = None
    min_time_difference = float('inf')
    
    # Find the nearest restaurant for the current customer
    for _, restaurant_row in restaurants_df.iterrows():
        restaurant_time = restaurant_row['date_y_time_y']
        time_difference = calculate_time_difference(customer_time, restaurant_time)
        if time_difference < min_time_difference:
            min_time_difference = time_difference
            nearest_restaurant = (restaurant_row['Unnamed: 0'], restaurant_row['date_y_time_y'])
    
    # Assign the nearest restaurant and its time to the current customer
    nearest_restaurants[customer_row['Unnamed: 0']] = nearest_restaurant

# Convert the nearest_restaurants dictionary to a DataFrame
nearest_restaurants_df = pd.DataFrame.from_dict(nearest_restaurants, orient='index', columns=['Nearest_Restaurant', 'Restaurant_Date_Time'])

# Merge the nearest restaurant information with the customer DataFrame
matched_customers_df = pd.merge(customers_df, nearest_restaurants_df, left_on='Unnamed: 0', right_index=True)

# Print the matched customers with their nearest restaurants, restaurant dates, and times
print("Matched Customers with Nearest Restaurants for Specific Dates:")
print(matched_customers_df)
