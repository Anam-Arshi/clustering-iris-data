# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load the Iris dataset from sklearn
iris = load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Target labels (species)

# Convert the dataset into a pandas DataFrame for easier handling
df = pd.DataFrame(X, columns=iris.feature_names)
df['Species'] = iris.target_names[y]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction to 2D (for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Perform K-means clustering (choose k=3 since there are 3 species in Iris)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the results of clustering using PCA components
plt.figure(figsize=(8, 6))

# Scatter plot using PCA components (2D plot)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis', s=100, alpha=0.6)

# Mark the centroids of the clusters
centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')

# Add labels and title
plt.title('K-means Clustering of Iris Dataset (PCA projection)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

# Print the cluster assignments for each sample
print("\nCluster Assignments:")
print(df[['Species', 'Cluster']])

# Print the K-means centroids in original space
print("\nCluster Centroids (in original feature space):")
print(kmeans.cluster_centers_)

