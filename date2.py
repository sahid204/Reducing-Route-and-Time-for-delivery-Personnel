import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load CSV file into a pandas DataFrame
df = pd.read_csv('output1.csv')

# Combine date and time into a single datetime column
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

# Select relevant features for clustering
features = ['datetime']

# Standardize the features (consider whether it's necessary)
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Specify the number of clusters (you may need to adjust this based on your data)
num_clusters = 2

# Perform k-medoids clustering
kmedoids = KMedoids(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmedoids.fit_predict(df[features])

# Visualize the clusters with a scatter plot
plt.scatter(df['datetime'], df['cluster'], c=df['cluster'], cmap='viridis', marker='o', s=50)
plt.xlabel('Datetime')
plt.ylabel('Cluster')
plt.title('Clusters based on Datetime (KMedoids)')
plt.show()
