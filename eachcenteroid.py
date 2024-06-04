import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

data = pd.read_csv("output1.csv")
data.head()

# Visualize data points
X = data[["latitude", "longitude"]]
plt.scatter(X["latitude"], X["longitude"], c="blue")
plt.xlabel("latitude")
plt.ylabel("longitude")
plt.show()

# Select K centroids from data points and assign them as red for representation
K = 3  # Change the number of centroids as needed
Centroids = X.sample(n=K)
plt.scatter(X["latitude"], X["longitude"], c="blue")
plt.scatter(Centroids["latitude"], Centroids["longitude"], c="red")
plt.xlabel("latitude")
plt.ylabel("longitude")
plt.show()

# Initialize variables for clustering
diff = 1
j = 0

while diff != 0:
    XD = X
    i = 1
    for index1, row_c in Centroids.iterrows():
        ED = []
        for index2, row_d in XD.iterrows():
            d1 = (row_c["latitude"] - row_d["latitude"])**2
            d2 = (row_c["longitude"] - row_d["longitude"])**2
            d = sqrt(d1 + d2)
            ED.append(d)
        X[i] = ED
        i = i + 1

    C = []
    for index, row in X.iterrows():
        min_dist = row[1]
        pos = 1
        for i in range(K):
            if row[i + 1] < min_dist:
                min_dist = row[i + 1]
                pos = i + 1
        C.append(pos)
    X["Cluster"] = C
    Centroids_new = X.groupby(["Cluster"]).mean()[["longitude", "latitude"]]
    if j == 0:
        diff = 1
        j = j + 1
    else:
        diff = (Centroids_new['longitude'] - Centroids['longitude']).sum() + (Centroids_new['latitude'] - Centroids['latitude']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["longitude", "latitude"]]

# Calculate the centroids for each cluster
cluster_centroids = []
for k in range(K):
    cluster_centroid = Centroids.iloc[k]
    print(f"Centroid for Cluster {k + 1}:")
    print(f"Latitude: {cluster_centroid['latitude']}, Longitude: {cluster_centroid['longitude']}")
    cluster_centroids.append(cluster_centroid)

# Specify the cluster number for which you want to find the nearest point
cluster_to_check = 2  # Change this to the cluster number you want to check

# Get the coordinates of the cluster's centroid
cluster_centroid = Centroids.iloc[cluster_to_check - 1]

# Filter the data for the specified cluster
cluster_data = X[X["Cluster"] == cluster_to_check]

# Calculate the Euclidean distances for all points in the cluster
distances = cluster_data.apply(lambda row: sqrt((row['latitude'] - cluster_centroid['latitude'])**2 + (row['longitude'] - cluster_centroid['longitude'])**2), axis=1)

# Find the index of the point with the minimum distance
nearest_point_index = distances.idxmin()

# Get the nearest point's coordinates
nearest_point = X.loc[nearest_point_index]

print("Nearest Point to Cluster", cluster_to_check, ":")
print("Latitude:", nearest_point['latitude'])
print("Longitude:", nearest_point['longitude'])

# The rest of your code remains unchanged for other clusters
