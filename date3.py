import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load CSV file into a pandas DataFrame
df = pd.read_csv('output1.csv')

# Combine date and time into a single datetime column
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d-%m-%Y %I:%M %p')

# Select relevant features for clustering
features = ['datetime']

# Standardize the features (consider whether it's necessary)
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

eps_value = 0.2  # Example value, adjust based on your data
min_samples_value = 5  # Example value, adjust based on your data # Adjust this based on your data

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
df['cluster'] = dbscan.fit_predict(df[features])

# Visualize the clusters with a scatter plot
plt.scatter(df['datetime'], df['cluster'], c=df['cluster'], cmap='viridis', marker='o', s=50)
plt.xlabel('Datetime')
plt.ylabel('Cluster')
plt.title('Clusters based on Datetime (DBSCAN)')
plt.show()
