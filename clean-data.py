# =========================================================
# Imports — manipulation et visualisation
# =========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# Imports — Machine Learning (scikit-learn)
# =========================================================
# Préparation des données
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Régression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor          # bonus
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier         # bonus
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report)

# Sauvegarde des modèles (pour les réutiliser dans Streamlit)
import joblib

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)

# =========================================================
# 🔗 Jeu de données (lu directement depuis Internet)
# =========================================================
URL_SPOTIFY = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
df = pd.read_csv(URL_SPOTIFY)

print('Dimensions initiales :', df.shape)
print("Colonnes :")
# On liste les types de chaque colonne pour identifier les colonnes catégorielles et numériques.
print(df.dtypes.to_frame('type'))
# On vérifie la présence de valeurs manquantes.
print(df.isna().sum().sort_values(ascending=False).to_frame('valeurs_manquantes'))
# On vérifie la présence de morceaux associés à plusieurs genres.
genres_par_morceau = df.groupby('track_id')['playlist_genre'].nunique()
# On affiche le nombre de morceaux uniques, le nombre de lignes dupliquées et le nombre de morceaux associés à plusieurs genres.
print('Morceaux uniques :', df['track_id'].nunique())
# On affiche le nombre de morceaux uniques, le nombre de lignes dupliquées et le nombre de morceaux associés à plusieurs genres.
print('Lignes dupliquées par track_id :', int(df['track_id'].duplicated().sum()))
print('Morceaux associés à plusieurs genres :', int((genres_par_morceau > 1).sum()))

df = df.drop_duplicates(subset='track_id', keep='first').copy()
df['duration_min'] = df['duration_ms'] / 60000
df['playlist_genre'] = df['playlist_genre'].astype('category')

cleaned_df = df.copy()  # <-- dataSet propre pour backup

print('Dimensions après déduplication :', df.shape)
# print(df.head())

audio_cols = ['danceability', 'energy', 'speechiness', 'acousticness',
              'instrumentalness', 'liveness', 'valence', 'loudness', 'tempo']
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
for col, ax in zip(audio_cols, axes.flat):
    sns.histplot(data=df, x=col, kde=True, ax=ax)
    ax.set_title(f'Distribution de {col}')
plt.tight_layout()
plt.show()

import os
os.makedirs("data", exist_ok=True)
df.to_csv("data/spotify_clean.csv", index=False)