from pathlib import Path

import pandas as pd
import streamlit as st

# Fichier à la racine du projet : pages/accueil.py -> ../data/spotify_clean.csv
RACINE = Path(__file__).resolve().parent.parent
DATA_PATH = RACINE / "data" / "spotify_clean.csv"

URL_SOURCE = "https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-01-21"
LIGNES_BRUTES, COLONNES_BRUTES = 32_833, 23   # dimensions du fichier d'origine


@st.cache_data
def charger_donnees() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


# Description des variables (le type, les manquants et les valeurs uniques sont calculés)
DESCRIPTIONS = {
    "track_id": "Identifiant unique du morceau",
    "track_name": "Titre du morceau",
    "track_artist": "Artiste du morceau",
    "track_popularity": "Popularité du morceau, de 0 à 100 (plus c'est haut, plus il est populaire)",
    "track_album_id": "Identifiant unique de l'album",
    "track_album_name": "Nom de l'album",
    "track_album_release_date": "Date de sortie de l'album",
    "playlist_name": "Nom de la playlist qui contient le morceau",
    "playlist_id": "Identifiant de la playlist",
    "playlist_genre": "Genre de la playlist : pop, rap, rock, latin, r&b ou edm (variable cible de la classification)",
    "playlist_subgenre": "Sous-genre de la playlist",
    "danceability": "Aptitude du morceau à la danse, de 0 (peu dansant) à 1 (très dansant)",
    "energy": "Intensité et activité perçues, de 0 à 1 (variable cible de la régression)",
    "key": "Tonalité estimée (0 = Do, 1 = Do♯/Ré♭, etc. ; -1 si non détectée)",
    "loudness": "Volume sonore moyen en décibels (dB), en général entre -60 et 0",
    "mode": "Mode du morceau : 1 = majeur, 0 = mineur",
    "speechiness": "Présence de paroles parlées, de 0 à 1 (proche de 1 : podcast, rap parlé)",
    "acousticness": "Probabilité que le morceau soit acoustique, de 0 à 1",
    "instrumentalness": "Probabilité que le morceau ne contienne pas de voix, de 0 à 1",
    "liveness": "Présence d'un public (enregistrement en direct), de 0 à 1",
    "valence": "Positivité musicale, de 0 (triste, sombre) à 1 (joyeux, enjoué)",
    "tempo": "Tempo estimé en battements par minute (BPM)",
    "duration_ms": "Durée du morceau en millisecondes",
    "duration_min": "Durée du morceau en minutes (variable créée : duration_ms / 60 000)",
    "release_year": "Année de sortie de l'album (variable créée à partir de la date de sortie)",
}

df = charger_donnees()

# =========================================================
st.title("🎵 Prédire l'énergie et le genre d'un morceau Spotify")
st.caption("Projet de Machine Learning · IRA4 · Esaip")

# =========================================================
# Contexte et objectif
# =========================================================
st.header("Contexte et objectif du projet")
st.markdown(
    """
    Ce projet a été réalisé dans le cadre du **cours de Machine Learning de
    M. Gbale**, en **IRA4 à l'Esaip**.

    Nous travaillons sur un jeu de données de morceaux Spotify, décrits par leurs
    caractéristiques audio (dansabilité, énergie, volume, tempo, etc.).
    """
)

st.markdown("**Problématique :** *peut-on prédire l'énergie d'un morceau et son genre "
            "musical à partir de ses seules caractéristiques audio ?*")

col_reg, col_clf = st.columns(2)
with col_reg:
    st.markdown(
        """
        #### 📈 Régression
        Prédire une valeur continue : l'**énergie** (`energy`) du morceau,
        entre 0 et 1.
        """
    )
with col_clf:
    st.markdown(
        """
        #### 🎯 Classification
        Prédire une catégorie : le **genre** (`playlist_genre`), parmi 6 classes
        (pop, rap, rock, latin, r&b, edm).
        """
    )

st.subheader("Membres du groupe")
membres = ["Ian Redor", "Aline Blanchouin", "Victor Henry", "Naël Crasson", "Mélissande Garçonnet"]
colonnes = st.columns(len(membres))
for col, nom in zip(colonnes, membres):
    col.markdown(f"**{nom}**")

st.divider()

# =========================================================
# Jeu de données
# =========================================================
st.header("Présentation du jeu de données")

st.markdown(
    f"""
    **Source :** jeu de données `spotify_songs` du projet communautaire
    [TidyTuesday]({URL_SOURCE}) (semaine du 21 janvier 2020), construit à partir de
    l'API de Spotify. Il contient des morceaux issus de playlists de six genres.
    """
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Lignes", f"{df.shape[0]:,}".replace(",", " "))
c2.metric("Colonnes", df.shape[1])
c3.metric("Genres", df["playlist_genre"].nunique())
c4.metric("Valeurs manquantes", int(df.isna().sum().sum()))

brutes = f"{LIGNES_BRUTES:,}".replace(",", " ")
st.markdown(
    f"""
    **Nettoyage effectué** (script `clean-data.py`) :
    - le fichier d'origine compte **{brutes} lignes et {COLONNES_BRUTES} colonnes**, car un
      même morceau peut figurer dans plusieurs playlists ;
    - suppression des doublons sur `track_id` (première occurrence conservée) ;
    - ajout de la colonne `duration_min` (durée en minutes).
    """
)

st.subheader("Aperçu des données")
st.dataframe(df.head(50))

# =========================================================
# Dictionnaire des variables
# =========================================================
st.subheader("Dictionnaire des variables")

dico = pd.DataFrame({
    "Variable": df.columns,
    "Description": [DESCRIPTIONS.get(c, "—") for c in df.columns],
    "Type": df.dtypes.astype(str).values,
    "Manquantes": df.isna().sum().values,
    "Valeurs uniques": df.nunique().values,
})
st.dataframe(dico, hide_index=True)