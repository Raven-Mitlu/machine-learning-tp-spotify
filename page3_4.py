import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title="Projet Spotify - Predictions", layout="wide")

# Masquer l'icône de lien sur les titres
st.markdown("""<style>a.header-anchor {display: none !important;}</style>""", unsafe_allow_html=True)

# ==========================================
# 1. CHARGEMENT DES DONNeES ET MODELES (CACHE)
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("./data/spotify_clean.csv")
    df['titre_complet'] = df['track_artist'] + " - " + df['track_name'] + " (Realite : " + df['playlist_genre'].str.upper() + ")"
    df = df.dropna(subset=['titre_complet'])
    return df.drop_duplicates(subset=['titre_complet'])

@st.cache_resource
def load_models():
    try:
        modele_reg = joblib.load("meilleur_modele_energie.joblib")
    except:
        try:
            modele_reg = joblib.load("modele_regression_energie.joblib")
        except:
            modele_reg = None 
        
    try:
        modele_clf = joblib.load("meilleur_modele_spotify.joblib")
    except:
        modele_clf = None
        
    try:
        modele_lineaire = joblib.load("modele_lineaire_energie.joblib")
    except:
        modele_lineaire = None
        
    return modele_reg, modele_clf, modele_lineaire

modele_reg, modele_clf, modele_lineaire = load_models()
df_spotify = load_data()

# ==========================================
# 2. NAVIGATION
# ==========================================
st.sidebar.title("Navigation Spotify")
page = st.sidebar.radio("Aller à la page :", [
    "Page 3 : Modele de regression", 
    "Page 4 : Modele de classification"
])

# ==========================================
# PAGE 3 : REGRESSION (ENERGIE)
# ==========================================
if page == "Page 3 : Modele de regression":
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

# ==========================================
# PAGE 4 : CLASSIFICATION (GENRES)
# ==========================================
elif page == "Page 4 : Modele de classification":
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