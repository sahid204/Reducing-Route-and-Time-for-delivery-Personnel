import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load CSV file into a pandas DataFrame (this is the original data used for clustering)
df = pd.read_csv('output1.csv')

# Combine date and time into a single datetime column
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d-%m-%Y %I:%M %p')  # Updated format

# Define features for clustering
numeric_features = ['latitude', 'longitude', 'datetime']

# Extract numeric features from the DataFrame
X = df[numeric_features].copy()  # Make a copy of the DataFrame using .copy()

# Ensure the 'datetime' column has the correct data type
X['datetime'] = pd.to_numeric(X['datetime'])

# Standardize numeric features
scaler = StandardScaler()
X[['latitude', 'longitude', 'datetime']] = scaler.fit_transform(X[['latitude', 'longitude', 'datetime']])

# Perform KMeans clustering
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X[['latitude', 'longitude', 'datetime']])

# Assume 'new_data' is a DataFrame with 'latitude', 'longitude', 'date', and 'time' for the new data point.
# Now, preprocess the new data to match the features used during fit time
new_data = pd.DataFrame({
    'latitude': [12.9500],
    'longitude': [77.6000],
    'date': ['26-01-2024'],
    'time': ['8:30 PM']
})

# Combine date and time into a single datetime column for new data
new_data['datetime'] = pd.to_numeric(pd.to_datetime(new_data['date'] + ' ' + new_data['time'], format='%d-%m-%Y %I:%M %p'))

# Standardize 'latitude', 'longitude', and 'datetime' for new data
new_data[['latitude', 'longitude', 'datetime']] = scaler.transform(new_data[['latitude', 'longitude', 'datetime']])

# Predict the cluster for the new data point
new_cluster_label = kmeans.predict(new_data[['latitude', 'longitude', 'datetime']])

# The 'new_cluster_label' variable now contains the predicted cluster for the new data point.
print("Predicted Cluster for Person Point:", new_cluster_label[0])
