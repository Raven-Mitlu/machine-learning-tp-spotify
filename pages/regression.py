import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# ==========================================
# 1. CHARGEMENT DES DONNeES ET MODELES (CACHE)
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("./data/spotify_clean.csv")
    df['titre_complet'] = df['track_artist'] + " - " + df['track_name'] + " (Realite : " + df['playlist_genre'].str.upper() + ")"
    df = df.dropna(subset=['titre_complet'])
    return df.drop_duplicates(subset=['titre_complet'])


from utils import charger_modele


@st.cache_resource
def load_models():
    modele_reg = charger_modele("modele_regression_energie.joblib")
    modele_clf = charger_modele("meilleur_modele_spotify.joblib")
    modele_lineaire = charger_modele("modele_lineaire_energie.joblib")
    return modele_reg, modele_clf, modele_lineaire

modele_reg, modele_clf, modele_lineaire = load_models()
df_spotify = load_data()

st.title("Prediction de l'energie de la musique")
    
with st.expander("Informations et performances du modele de regression"):
    st.markdown("""
    - **Algorithme retenu :** Forêt Aleatoire (RandomForestRegressor)
    - **Variable cible :** `energy` (energie du morceau, entre 0 et 1)
    - **Variables explicatives :** Attributs audio (`danceability`, `loudness`, `tempo`, `acousticness`, `valence`, `speechiness`, `instrumentalness`, `liveness`, `duration_min`, `key`, `mode`)
    - **Preparation & Encodage :** Suppression des doublons, normalisation/standardisation des variables numériques si nécessaire.
    - **Parametres du modèle :** `random_state=42`, n_estimators par défaut (forêt aléatoire).
    - **Decoupage :** Train / Test (80% / 20%, `random_state=42`)
    """)
    
    st.markdown("#### Tableau comparatif des modeles testes :")
    df_perf_reg = pd.DataFrame({
        "Modele": ["Forêt Aleatoire", "Lineaire + categories", "Lineaire numerique", "Lineaire simple"],
        "R² Test": [0.72, 0.67, 0.63, 0.46],
        "MAE Test": [0.075, 0.083, 0.088, 0.107],
        "RMSE Test": [0.098, 0.105, 0.111, 0.134]
    })
    st.table(df_perf_reg)
    
    st.markdown("#### Graphiques d'evaluation :")
    try:
        st.image("./image/reelles_vs_predites.png", caption="Valeurs reelles vs Valeurs predites", use_container_width=True)
    except:
        st.info("Image 'reelles_vs_predites.png' non trouvée (optionnel pour la soutenance).")
    try:
        st.image("./image/coefficients_lineaire.png", caption="Coefficients du modele lineaire", use_container_width=True)
    except:
        st.info("Image 'coefficients_lineaire.png' non trouvée (optionnel pour la soutenance).")

st.markdown("### Modifiez les attributs pour estimer l'energie :")

with st.form("form_regression"):
    col1, col2 = st.columns(2)
    with col1:
        loudness = st.slider("Volume (loudness en dB)", min_value=-60.0, max_value=0.0, value=-5.0)
        danceability = st.slider("Danceability", min_value=0.0, max_value=1.0, value=0.7)
        tempo = st.slider("Tempo (BPM)", min_value=50.0, max_value=220.0, value=120.0)
    with col2:
        acousticness = st.slider("Acousticness", min_value=0.0, max_value=1.0, value=0.1)
        valence = st.slider("Valence (Positivite)", min_value=0.0, max_value=1.0, value=0.5)
    
    submit_reg = st.form_submit_button("Predire l'energie")
    
if submit_reg:
    if modele_reg is not None:
        colonnes_reg = [
            'danceability', 'energy', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'loudness',
            'tempo', 'duration_min', 'key', 'mode'
        ]
        ligne_ref = df_spotify.iloc[0].copy()
        ligne_ref['danceability'] = danceability
        ligne_ref['acousticness'] = acousticness
        ligne_ref['valence'] = valence
        ligne_ref['loudness'] = loudness
        ligne_ref['tempo'] = tempo
        
        X_reg_user = pd.DataFrame([ligne_ref[colonnes_reg].drop(labels=['energy'], errors='ignore').values], columns=[c for c in colonnes_reg if c != 'energy'])
        prediction_energie = modele_reg.predict(X_reg_user)[0]
    else:
        prediction_energie = 0.5 
        st.warning("Le fichier du modèle de régression est introuvable dans le dossier.")
    
    st.markdown("---")
    st.markdown("### Resultat de l'analyse")
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction_energie,
        title={'text': "Dynamisme percu (Energy)", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "#1DB954"},
            'steps': [
                {'range': [0, 0.4], 'color': "#2A2A2A"},
                {'range': [0.4, 0.7], 'color': "#535353"},
                {'range': [0.7, 1.0], 'color': "#B3B3B3"}
            ]
        }
    ))
    fig_gauge.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    st.plotly_chart(fig_gauge, use_container_width=True)