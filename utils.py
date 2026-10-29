import pandas as pd
import streamlit as st
 
DATA_PATH = "data/spotify_clean.csv"
 
 
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

import zipfile
from pathlib import Path

import joblib
import streamlit as st

RACINE = Path(__file__).resolve().parent  # dossier qui contient app.py


def charger_modele(nom_joblib: str):
    """Charge un modèle .joblib, en dézippant l'archive du même nom si besoin."""
    chemin = RACINE / nom_joblib

    if not chemin.exists():
        archive = chemin.with_suffix(".zip")
        if archive.exists():
            with st.spinner(f"Décompression de {archive.name}..."):
                with zipfile.ZipFile(archive) as z:
                    z.extractall(RACINE)
            # si l'archive a créé un sous-dossier, on retrouve le fichier
            if not chemin.exists():
                trouves = list(RACINE.rglob(nom_joblib))
                if trouves:
                    chemin = trouves[0]

    if not chemin.exists():
        st.warning(f"Modèle introuvable : {nom_joblib} (ni le fichier, ni son archive .zip).")
        return None

    try:
        return joblib.load(chemin)
    except Exception as e:
        st.warning(f"Échec du chargement de {nom_joblib} : {e}")
        return None