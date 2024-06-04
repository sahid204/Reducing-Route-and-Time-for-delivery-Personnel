import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load your data from "output1.csv"
data = pd.read_csv("out_1.csv")

# Select the features for clustering
X = data[["latitude", "longitude"]]

# Visualize the data points
plt.scatter(X["latitude"], X["longitude"], c="blue")
plt.xlabel("latitude")
plt.ylabel("longitude")
plt.show()

# Number of clusters (you can adjust this)
K = 3

# Initialize centroids by randomly selecting data points
Centroids = X.sample(n=K)

# Visualize the data points and initial centroids
plt.scatter(X["latitude"], X["longitude"], c="blue")
plt.scatter(Centroids["latitude"], Centroids["longitude"], c="red")
plt.xlabel("latitude")
plt.ylabel("longitude")
plt.show()

# Initialize variables for the loop
diff = 1
j = 0

while diff != 0:
    XD = X
    i = 1
    for index1, row_c in Centroids.iterrows():
        ED = []
        for index2, row_d in XD.iterrows():
            d1 = (row_c["latitude"] - row_d["latitude"]) ** 2
            d2 = (row_c["longitude"] - row_d["longitude"]) ** 2
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
        diff = (
            (Centroids_new['longitude'] - Centroids['longitude']).sum()
            + (Centroids_new['latitude'] - Centroids['latitude']).sum()
        )
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["longitude", "latitude"]] 

# Assign colors to clusters
color = ['blue', 'green', 'cyan']

# Plot the clustered data points
for k in range(K):
    data = X[X["Cluster"] == k + 1]
    plt.scatter(data["latitude"], data["longitude"], c=color[k])

# Plot the cluster centroids in red
plt.scatter(Centroids["latitude"], Centroids["longitude"], c='red')
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.show()

# Calculate Silhouette Score
silhouette_avg = silhouette_score(X, X["Cluster"])

print(f"Silhouette Score: {silhouette_avg:.2f}")
