# ================================
# Simple Recommendation System
# ================================

import pandas as pd

# Load clustered dataset
df = pd.read_csv("clustered_spotify_songs.csv")

def recommend_songs(cluster_number):
    recommendations = df[df['Cluster'] == cluster_number]
    return recommendations[['track_name','playlist_genre','popularity']].head(10)

# Example
print(recommend_songs(2))
