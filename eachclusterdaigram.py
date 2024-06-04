import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
from math import sqrt
import networkx as nx

# Read the data from the CSV file
data = pd.read_csv("out_1.csv")

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

# Calculate the overall centroid by taking the mean of cluster centroids
overall_centroid = pd.DataFrame(cluster_centroids).mean()
print("\nOverall Centroid:")
print(f"Latitude: {overall_centroid['latitude']}, Longitude: {overall_centroid['longitude']}")

# Create a NetworkX graph
G = nx.Graph()

# Add the overall centroid as a node to the graph
G.add_node("Overall Centroid", latitude=overall_centroid['latitude'], longitude=overall_centroid['longitude'])

# Add each cluster centroid as a node and calculate the distance to the overall centroid
for k in range(K):
    cluster_centroid = Centroids.iloc[k]
    node_name = f"Cluster {k + 1} Centroid"
    latitude = cluster_centroid['latitude']
    longitude = cluster_centroid['longitude']
    G.add_node(node_name, latitude=latitude, longitude=longitude)
    distance_to_overall_centroid = sqrt((overall_centroid['latitude'] - latitude)**2 + (overall_centroid['longitude'] - longitude)**2)
    G.add_edge("Overall Centroid", node_name, weight=distance_to_overall_centroid)

# Ask the user for the target cluster
target_cluster = "Cluster 2 Centroid"

# ... (previous code)

# Find the shortest path from the overall centroid to the specified cluster
if target_cluster in G.nodes:
    shortest_path = nx.shortest_path(G, source="Overall Centroid", target=target_cluster, weight="weight")
    path_length = nx.shortest_path_length(G, source="Overall Centroid", target=target_cluster, weight="weight")

    # Create a layout for the graph nodes
    pos = {node: (G.nodes[node]['longitude'], G.nodes[node]['latitude']) for node in G.nodes}

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8)) 

    # Draw the nodes and edges
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=2000, node_color="lightblue", font_size=10)

    # Highlight the shortest path by drawing it with a different color
    shortest_path_edges = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]
    edge_colors = ['blue' if edge in shortest_path_edges else 'gray' for edge in G.edges()]
    nx.draw(G, pos, ax=ax, edgelist=shortest_path_edges, edge_color=edge_colors, width=2, node_size=2000, node_color="lightblue", font_size=10)

    # Add labels for the nodes
    labels = {node: node for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color="black")

    # Annotate the diagram with the distance
    for i in range(len(shortest_path) - 1):
        edge = (shortest_path[i], shortest_path[i + 1])
        x = (pos[edge[0]][0] + pos[edge[1]][0]) / 2
        y = (pos[edge[0]][1] + pos[edge[1]][1]) / 2
        distance = G.edges[edge]["weight"]
        ax.annotate(f"Distance: {distance:.2f}", (x, y), fontsize=10, color='blue', horizontalalignment='center')

    # Display the graph
    plt.title("Shortest Path to " + target_cluster)
    plt.show()
else:
    print(f"Node {target_cluster} not found in the graph.")
