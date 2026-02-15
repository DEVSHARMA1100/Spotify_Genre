# ================================
# Spotify Genre Segmentation Model
# ================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("spotify_songs.csv")

# -----------------------------
# Data Preprocessing
# -----------------------------

# Remove duplicates
df = df.drop_duplicates()

# Fill missing values
df = df.fillna(df.mean(numeric_only=True))

# Select audio features
features = df[['danceability','energy','loudness','speechiness',
               'acousticness','instrumentalness',
               'liveness','valence','tempo']]

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# -----------------------------
# Elbow Method
# -----------------------------

inertia = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.plot(range(1,11), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# -----------------------------
# Apply KMeans
# -----------------------------

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# -----------------------------
# PCA Visualization
# -----------------------------

pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

df['PCA1'] = pca_features[:,0]
df['PCA2'] = pca_features[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set1')
plt.title("Cluster Visualization")
plt.show()

# Save updated dataset
df.to_csv("clustered_spotify_songs.csv", index=False)

print("Model executed successfully.")
