import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# ==========================================
# 1. CHARGEMENT DES DONNeES ET MODELES (CACHE)
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils import charger_modele


@st.cache_data
def load_data():
    df = pd.read_csv("./data/spotify_clean.csv")

    # recrée release_year si le CSV ne la contient pas
    if "release_year" not in df.columns:
        df["release_year"] = pd.to_numeric(
            df["track_album_release_date"].astype(str).str[:4], errors="coerce"
        )
    df = df.dropna(subset=["release_year", "track_artist", "track_name"])
    df["release_year"] = df["release_year"].astype(int)

    df["titre_complet"] = (df["track_artist"] + " - " + df["track_name"]
                           + " (Realite : " + df["playlist_genre"].str.upper() + ")")
    return df.drop_duplicates(subset=["titre_complet"])


@st.cache_resource
def load_models():
    modele_reg = charger_modele("modele_regression_energie.joblib")
    modele_clf = charger_modele("meilleur_modele_spotify.joblib")
    modele_lineaire = charger_modele("modele_lineaire_energie.joblib")
    return modele_reg, modele_clf, modele_lineaire


modele_reg, modele_clf, modele_lineaire = load_models()
df_spotify = load_data()

modele_reg, modele_clf, modele_lineaire = load_models()
df_spotify = load_data()

st.title("Prediction du genre de la musique")
    
with st.expander("Informations et performances du modele de classification"):
    st.markdown("""
    - **Algorithme retenu :** Regression Logistique / KNN / Forêt Aléatoire
    - **Variable cible :** `playlist_genre` (6 classes : edm, latin, pop, r&b, rap, rock)
    - **Variables explicatives :** `danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `loudness`, `tempo`, `duration_min`, `key`, `mode`, `release_year`
    - **Preparation & Parametres :** Standardisation avec `StandardScaler` (fit sur le train uniquement pour éviter le data leakage). Paramètres clés : `max_iter=1000` (Régression Logistique), test de plusieurs valeurs de `k` (KNN).
    - **Metrique principale :** Le **F1-Score (Macro)** a ete privilegie pour compenser les desequilibres entre les genres musicaux.
    """)
    
    st.markdown("#### Tableau comparatif des performances (Train / Test) :")
    df_perf_clf = pd.DataFrame({
        "Modèle": ["Régression Logistique", "KNN", "Forêt Aléatoire"],
        "Accuracy": [0.544, 0.520, 0.610],
        "Précision (Macro)": [0.540, 0.515, 0.605],
        "Rappel (Macro)": [0.538, 0.510, 0.600],
        "F1-Score (Macro)": [0.540, 0.512, 0.602]
    })
    st.table(df_perf_clf)
    
    st.markdown("#### Matrice de confusion (6x6) :")
    st.write("Analyse des réussites et des confusions entre les 6 genres musicaux :")
    try:
        st.image("./image/matrice_confusion.png", caption="Matrice de confusion - Modele de classification", use_container_width=True)
    except:
        st.warning("Fichier 'matrice_confusion.png' non trouvé dans le dossier.")

st.markdown("### Test sur un morceau existant ou en modifiant les attributs :")

liste_morceaux = ["Selectionnez un titre..."] + df_spotify['titre_complet'].tolist()
morceau_choisi = st.selectbox("Selectionnez un titre pour charger ses attributs :", liste_morceaux)

def_vals = {
    'danceability': 0.75, 'energy': 0.80, 'speechiness': 0.05, 'acousticness': 0.10,
    'instrumentalness': 0.0, 'liveness': 0.15, 'valence': 0.50, 'loudness': -5.0,
    'tempo': 120.0, 'duration_min': 3.5, 'key': 5, 'mode': 1, 'release_year': 2019
}

if morceau_choisi != "Selectionnez un titre...":
    track_data = df_spotify[df_spotify['titre_complet'] == morceau_choisi].iloc[0]
    for col in def_vals.keys():
        def_vals[col] = float(track_data[col])
        
st.markdown("### Ou modifiez les attributs manuellement :")

with st.form("form_classification"):
    col1, col2 = st.columns(2)
    with col1:
        danceability_clf = st.slider("Danceability", 0.0, 1.0, float(def_vals['danceability']))
        energy_clf = st.slider("Energy", 0.0, 1.0, float(def_vals['energy']))
        speechiness_clf = st.slider("Speechiness", 0.0, 1.0, float(def_vals['speechiness']))
        acousticness_clf = st.slider("Acousticness", 0.0, 1.0, float(def_vals['acousticness']))
        instrumentalness_clf = st.slider("Instrumentalness", 0.0, 1.0, float(def_vals['instrumentalness']))
        liveness_clf = st.slider("Liveness", 0.0, 1.0, float(def_vals['liveness']))
        valence_clf = st.slider("Valence", 0.0, 1.0, float(def_vals['valence']))
    with col2:
        loudness_clf = st.slider("Volume (loudness en dB)", -60.0, 0.0, float(def_vals['loudness']))
        tempo_clf = st.slider("Tempo (BPM)", 50.0, 220.0, float(def_vals['tempo']))
        duration_min = st.slider("Duree (minutes)", 1.0, 10.0, float(def_vals['duration_min']))
        key_clf = st.slider("Key (Tonalite)", 0, 11, int(def_vals['key']))
        mode_clf = st.selectbox("Mode", [1, 0], index=0 if int(def_vals['mode']) == 1 else 1, format_func=lambda x: "Majeur" if x == 1 else "Mineur")
        release_year_clf = st.slider("Annee de sortie", 1950, 2024, int(def_vals['release_year']))
        
    submit_clf = st.form_submit_button("Deviner le genre musical")

if modele_clf is not None and (submit_clf or morceau_choisi != "Selectionnez un titre..."):
    colonnes_modele = [
        'danceability', 'energy', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'loudness',
        'tempo', 'duration_min', 'key', 'mode', 'release_year'
    ]
    
    X_user = pd.DataFrame([[
        danceability_clf, energy_clf, speechiness_clf, acousticness_clf,
        instrumentalness_clf, liveness_clf, valence_clf, loudness_clf,
        tempo_clf, duration_min, key_clf, mode_clf, release_year_clf
    ]], columns=colonnes_modele)
            
    probas_reelles = modele_clf.predict_proba(X_user)[0]
    genres = modele_clf.classes_
            
    index_gagnant = np.argmax(probas_reelles)
    genre_gagnant = genres[index_gagnant]
            
    st.markdown("---")
    st.success(f"La musique est considere comme etant du : **{genre_gagnant.upper()}**")
            
    couleurs = ['#535353'] * len(genres)
    couleurs[index_gagnant] = '#1DB954'
            
    fig_bar = go.Figure(go.Bar(
        x=probas_reelles * 100, 
        y=[g.upper() for g in genres],
        orientation='h',
        marker_color=couleurs,
        text=[f"{p*100:.1f}%" for p in probas_reelles], 
        textposition='auto'
    ))
    
    fig_bar.update_layout(
        title="Probabilites par genre musical",
        xaxis=dict(title="Probabilite (%)", range=[0, 100]),
        yaxis=dict(title="Genres"),
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)