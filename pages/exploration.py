import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from utils import load_data

sns.set_theme(style="whitegrid")

df = load_data()

AUDIO_COLS = ["danceability", "energy", "speechiness", "acousticness",
              "instrumentalness", "liveness", "valence", "loudness", "tempo"]

st.title("📊 Exploration des données")

# =========================================================
# Filtres (sidebar) : mettent à jour tous les graphiques
# =========================================================
st.sidebar.header("Filtres")

genres = sorted(df["playlist_genre"].dropna().unique())
genres_choisis = st.sidebar.multiselect("Genres", genres, default=genres)

pmin, pmax = int(df["track_popularity"].min()), int(df["track_popularity"].max())
plage_pop = st.sidebar.slider("Popularité", pmin, pmax, (pmin, pmax))

filtered = df[
    df["playlist_genre"].isin(genres_choisis)
    & df["track_popularity"].between(*plage_pop)
]

if filtered.empty:
    st.warning("Aucun titre ne correspond aux filtres sélectionnés.")
    st.stop()

# =========================================================
# Indicateurs clés
# =========================================================
st.header("Indicateurs clés")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Titres", f"{len(filtered):,}".replace(",", " "),
          delta=f"{len(filtered) - len(df):,}".replace(",", " "))
c2.metric("Popularité moyenne", f"{filtered['track_popularity'].mean():.1f}")
c3.metric("Durée moyenne", f"{filtered['duration_min'].mean():.2f} min")
c4.metric("Artistes distincts", filtered["track_artist"].nunique())

st.divider()
st.header("Visualisations")

# =========================================================
# 1. Distribution des 9 caractéristiques audio (ta grille 3x3)
# =========================================================
st.subheader("1. Distribution des caractéristiques audio")
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
for col, ax in zip(AUDIO_COLS, axes.flat):
    sns.histplot(data=filtered, x=col, kde=True, ax=ax)
    ax.set_title(f"Distribution de {col}")
plt.tight_layout()
st.pyplot(fig)
st.markdown(
    "**Commentaire :** décris ici ce que tu observes : quelles variables sont "
    "symétriques, lesquelles sont très asymétriques (ex. `instrumentalness`, "
    "`speechiness`), et ce que ça implique pour la suite."
)

# =========================================================
# 2. Répartition des genres
# =========================================================
st.subheader("2. Répartition des titres par genre")
fig, ax = plt.subplots(figsize=(8, 4))
ordre = filtered["playlist_genre"].value_counts().index
sns.countplot(data=filtered, x="playlist_genre", order=ordre, ax=ax)
ax.set_xlabel("Genre")
ax.set_ylabel("Nombre de titres")
st.pyplot(fig)
st.markdown("**Commentaire :** les genres sont-ils équilibrés ? Lequel domine ?")

# =========================================================
# 3. Popularité selon le genre
# =========================================================
st.subheader("3. Popularité selon le genre")
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=filtered, x="playlist_genre", y="track_popularity", ax=ax)
ax.set_xlabel("Genre")
ax.set_ylabel("Popularité")
st.pyplot(fig)
st.markdown("**Commentaire :** quel genre est le plus populaire en médiane ? Où sont les écarts ?")

# =========================================================
# 4. Relation entre deux caractéristiques (choix interactif)
# =========================================================
st.subheader("4. Relation entre deux caractéristiques audio")
cx, cy = st.columns(2)
x_var = cx.selectbox("Axe X", AUDIO_COLS, index=1)
y_var = cy.selectbox("Axe Y", AUDIO_COLS, index=7)
echantillon = filtered.sample(min(len(filtered), 3000), random_state=0)
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=echantillon, x=x_var, y=y_var, hue="playlist_genre",
                alpha=0.5, s=18, ax=ax)
st.pyplot(fig)
st.caption("Échantillon de 3 000 titres maximum pour garder un affichage fluide.")
st.markdown("**Commentaire :** y a-t-il une relation visible entre ces deux variables ? Des groupes par genre ?")

# =========================================================
# 5. Matrice de corrélation
# =========================================================
st.subheader("5. Corrélations entre variables")
cols_corr = AUDIO_COLS + ["track_popularity", "duration_min"]
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(filtered[cols_corr].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax)
st.pyplot(fig)
st.markdown(
    "**Commentaire :** quelles variables sont fortement corrélées "
    "(ex. `energy` et `loudness`) ? Laquelle est la plus liée à la popularité ?"
)

# =========================================================
# 6. Profil moyen par genre (choix interactif)
# =========================================================
st.subheader("6. Profil moyen par genre")
var = st.selectbox("Caractéristique", AUDIO_COLS, key="profil")
moyennes = (filtered.groupby("playlist_genre", observed=True)[var]
            .mean().sort_values(ascending=False))
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=moyennes.index, y=moyennes.values, ax=ax)
ax.set_xlabel("Genre")
ax.set_ylabel(f"Moyenne de {var}")
st.pyplot(fig)
st.markdown("**Commentaire :** quels genres se distinguent sur cette caractéristique ?")