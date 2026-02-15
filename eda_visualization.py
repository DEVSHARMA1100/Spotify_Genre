# ================================
# Exploratory Data Analysis
# ================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("spotify_songs.csv")

# -----------------------------
# Popularity Distribution
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df['popularity'], bins=30, kde=True)
plt.title("Popularity Distribution")
plt.show()

# -----------------------------
# Songs per Playlist Genre
# -----------------------------
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='playlist_genre')
plt.xticks(rotation=45)
plt.title("Songs per Playlist Genre")
plt.show()

# -----------------------------
# Danceability vs Energy
# -----------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='danceability', y='energy', hue='playlist_genre')
plt.title("Danceability vs Energy")
plt.show()

# -----------------------------
# Correlation Matrix
# -----------------------------
plt.figure(figsize=(12,8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

print("EDA completed successfully.")
