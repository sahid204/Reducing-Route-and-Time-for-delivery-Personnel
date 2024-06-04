'''import pandas as pd

# Specify the input CSV file and output CSV file names
input_csv_file = 'Bangaloredata.csv'
output_csv_file = 'restaurantoutput1.csv'

# Read the input CSV file into a DataFrame
df = pd.read_csv(input_csv_file)

# Specify the columns containing latitude and longitude
latitude_column = 'Latitude'  # Replace with the actual column name in your CSV
longitude_column = 'Longitude'  # Replace with the actual column name in your CSV

# Remove duplicates based on latitude and longitude columns
df_no_duplicates = df.drop_duplicates(subset=[latitude_column, longitude_column])

# Write the DataFrame with duplicates removed to the output CSV file
df_no_duplicates.to_csv(output_csv_file, index=False)

# Print a summary of the operation
print(f"Removed {len(df) - len(df_no_duplicates)} duplicate records based on latitude and longitude.")


print(f"Saved unique records to '{output_csv_file}'.")'''

import pandas as pd
customer_clusters = pd.read_csv('output1.csv')
restaurant_clusters = pd.read_csv('restaurantoutput1.csv')

print("Customer Clusters Columns:", customer_clusters.columns)
print("Restaurant Clusters Columns:", restaurant_clusters.columns)
