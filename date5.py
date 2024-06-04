import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
from sklearn.pipeline import Pipeline

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

# Create a Birch clustering model
branching_factor = 50
threshold = 0.5
n_clusters = 3
birch = Birch(branching_factor=branching_factor, threshold=threshold, n_clusters=n_clusters)

# Create a standard scaler for numeric features
scaler = StandardScaler()

# Create a pipeline for clustering
pipeline = Pipeline([
    ('scaler', scaler),
    ('birch', birch),
])

# Fit the pipeline to the original data
pipeline.fit(X)

# Assume 'new_data' is a DataFrame with 'latitude', 'longitude', 'date', and 'time' for the new data point.
# Now, predict the cluster for the new data point
new_data = pd.DataFrame({
    'latitude': [12.9500],
    'longitude': [77.6000],
    'date': ['22-01-2024'],
    'time': ['1:30 PM']
})

new_data['datetime'] = pd.to_datetime(new_data['date'] + ' ' + new_data['time'], format='%d-%m-%Y %I:%M %p')  # Updated format
new_data['datetime'] = pd.to_numeric(new_data['datetime'])
new_cluster_label = pipeline.predict(new_data[['latitude', 'longitude', 'datetime']])

# The 'new_cluster_label' variable now contains the predicted cluster for the new data point.
print("Predicted Cluster for person Point:", new_cluster_label[0])
