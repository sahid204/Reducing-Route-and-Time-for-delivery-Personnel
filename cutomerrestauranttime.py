import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load assignments data from CSV file
assignments_df = pd.read_csv('assignments.csv')

# Function to convert time string to seconds since midnight
def time_to_seconds(time_str):
    hour, minute, second = map(int, time_str.split(':'))
    return hour * 3600 + minute * 60 + second

# Preprocess time values
assignments_df['Customer_Time_Seconds'] = assignments_df['Customer_Time'].apply(time_to_seconds)

# Combine customer and restaurant data into one DataFrame
combined_df = pd.concat([assignments_df[['Customer_Latitude', 'Customer_Longitude', 'Customer_Time_Seconds']],
                         assignments_df[['Restaurant_Latitude', 'Restaurant_Longitude', 'Restaurant_Time']].rename(columns={
                             'Restaurant_Latitude': 'Customer_Latitude',
                             'Restaurant_Longitude': 'Customer_Longitude',
                             'Restaurant_Time': 'Customer_Time'
                         })], ignore_index=True)

# Drop rows with missing values
combined_df.dropna(inplace=True)

# Cluster the combined data using K-means
kmeans = KMeans(n_clusters=2)  # Adjust the number of clusters as needed
combined_df['Cluster'] = kmeans.fit_predict(combined_df[['Customer_Latitude', 'Customer_Longitude', 'Customer_Time_Seconds']])

# Separate customer and restaurant data
customers = combined_df.iloc[:len(assignments_df)]
restaurants = combined_df.iloc[len(assignments_df):]

# Plot clustered data
plt.figure(figsize=(10, 8))
sns.scatterplot(data=customers, x='Customer_Longitude', y='Customer_Latitude', hue='Cluster', palette='coolwarm', label='Customers')
sns.scatterplot(data=restaurants, x='Customer_Longitude', y='Customer_Latitude', color='black', label='Restaurants', marker='^')
plt.title('Customers and Restaurants Clusters')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()
