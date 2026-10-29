import pandas as pd
import streamlit as st

# =========================================================
# Résultats des modèles (repris de vos pages Régression / Classification)
# À modifier ici si vos chiffres changent
# =========================================================
PERF_REG = pd.DataFrame({
    "Modèle": ["Forêt aléatoire", "Linéaire + catégories", "Linéaire numérique", "Linéaire simple"],
    "R² (test)": [0.72, 0.67, 0.63, 0.46],
    "MAE (test)": [0.075, 0.083, 0.088, 0.107],
    "RMSE (test)": [0.098, 0.105, 0.111, 0.134],
})

PERF_CLF = pd.DataFrame({
    "Modèle": ["Régression logistique", "KNN", "Forêt aléatoire"],
    "Accuracy": [0.544, 0.520, 0.610],
    "Précision (macro)": [0.540, 0.515, 0.605],
    "Rappel (macro)": [0.538, 0.510, 0.600],
    "F1-score (macro)": [0.540, 0.512, 0.602],
})

NB_GENRES = 6
HASARD = 1 / NB_GENRES

best_reg = PERF_REG.iloc[0]
base_reg = PERF_REG.iloc[-1]      # modèle de référence : linéaire simple
best_clf = PERF_CLF.iloc[2]
base_clf = PERF_CLF.iloc[0]       # modèle de référence : régression logistique

# =========================================================
st.title("🏁 Conclusion")

# =========================================================
# Réponse à la problématique
# =========================================================
st.header("Réponse à la problématique")
st.markdown(
    f"""
    **Problématique :** *peut-on prédire l'énergie d'un morceau (régression) et son
    genre musical (classification) à partir de ses caractéristiques audio ?*

    **Réponse :** oui, avec des niveaux de réussite très différents selon la tâche.

    - **L'énergie est bien prédite.** La forêt aléatoire atteint un R² de
      **{best_reg['R² (test)']:.2f}** et une erreur moyenne de **{best_reg['MAE (test)']:.3f}**
      sur une échelle de 0 à 1, bien mieux que le modèle linéaire simple
      (R² de {base_reg['R² (test)']:.2f}).
    - **Le genre est prédit nettement mieux que le hasard, mais sans fiabilité
      suffisante.** Avec {NB_GENRES} classes, le hasard donnerait environ
      **{HASARD:.0%}** de bonnes réponses. La forêt aléatoire atteint
      **{best_clf['Accuracy']:.1%}** d'accuracy et **{best_clf['F1-score (macro)']:.2f}**
      de F1 macro : les caractéristiques audio contiennent une vraie information
      sur le genre, mais elles ne suffisent pas à le déterminer.
    - Dans les deux cas, le **modèle non linéaire (forêt aléatoire) fait mieux** que
      les modèles linéaires ou basés sur la distance : les relations entre
      caractéristiques audio, énergie et genre ne sont pas linéaires.
    """
)

# =========================================================
# Résultats clés
# =========================================================
st.subheader("Résultats clés")
tab_reg, tab_clf = st.tabs(["Régression : prédire l'énergie", "Classification : prédire le genre"])

with tab_reg:
    st.caption("Meilleur modèle : forêt aléatoire. La variation affichée est l'écart avec le modèle linéaire simple.")
    c1, c2, c3 = st.columns(3)
    c1.metric("R² (test)", f"{best_reg['R² (test)']:.2f}",
              delta=f"{best_reg['R² (test)'] - base_reg['R² (test)']:+.2f}")
    c2.metric("MAE (test)", f"{best_reg['MAE (test)']:.3f}",
              delta=f"{best_reg['MAE (test)'] - base_reg['MAE (test)']:+.3f}",
              delta_color="inverse")
    c3.metric("RMSE (test)", f"{best_reg['RMSE (test)']:.3f}",
              delta=f"{best_reg['RMSE (test)'] - base_reg['RMSE (test)']:+.3f}",
              delta_color="inverse")
    st.dataframe(PERF_REG, hide_index=True)
    st.markdown(
        f"""
        **Commentaire :** la forêt aléatoire explique environ {best_reg['R² (test)']:.0%} de la
        variance de l'énergie et se trompe en moyenne de {best_reg['MAE (test)']:.3f} point
        sur une échelle de 0 à 1. Les modèles linéaires progressent à mesure qu'on leur
        ajoute des variables (de {base_reg['R² (test)']:.2f} à {PERF_REG.iloc[1]['R² (test)']:.2f}
        de R²), mais restent derrière la forêt aléatoire.
        """
    )

with tab_clf:
    st.caption("Meilleur modèle : forêt aléatoire. La variation affichée est l'écart avec la régression logistique.")
    c1, c2, c3, c4 = st.columns(4)
    for col, nom in zip((c1, c2, c3, c4),
                        ("Accuracy", "Précision (macro)", "Rappel (macro)", "F1-score (macro)")):
        col.metric(nom, f"{best_clf[nom]:.3f}", delta=f"{best_clf[nom] - base_clf[nom]:+.3f}")
    st.dataframe(PERF_CLF, hide_index=True)
    st.markdown(
        f"""
        **Commentaire :** la forêt aléatoire est le meilleur modèle sur les quatre
        métriques. Comme les genres sont déséquilibrés, c'est le **F1-score macro**
        ({best_clf['F1-score (macro)']:.2f}) qui compte le plus : il donne le même poids
        à chaque genre. Le score reste loin de 1, ce qui reflète le chevauchement
        des genres sur le plan audio.

        Les confusions les plus fréquentes sont entre **[à compléter d'après la matrice
        de confusion de la page Classification, par exemple : pop / r&b / rap]**.
        """
    )

st.divider()

# =========================================================
# Limites
# =========================================================
st.header("Limites des modèles")
st.markdown(
    f"""
    **Données**
    - **L'étiquette de genre vient de la playlist**, pas du morceau lui-même
      (`playlist_genre`). Les doublons ont été supprimés en gardant la première
      occurrence : un titre présent dans plusieurs playlists n'a donc qu'un seul genre.
    - **Jeu de données figé** : il date de 2020 et ne reflète ni les tendances actuelles
      ni les genres récents.
    - **Genres qui se chevauchent** : certains styles partagent des profils audio très
      proches, ce qui plafonne la performance quel que soit le modèle.

    **Modèles**
    - **Classification** : avec {best_clf['Accuracy']:.1%} d'accuracy, environ 4 morceaux sur 10
      sont mal classés. Le modèle ne peut pas servir à une décision fiable.
    - **Régression** : l'énergie est probablement très liée au volume (`loudness`).
      Une partie de la performance vient d'une seule variable ; la matrice de
      corrélation de la page Exploration permet de le vérifier.
    - **Évaluation** : un seul découpage train/test (80 % / 20 %) a été utilisé, sauf si
      vous avez fait de la validation croisée. Les scores peuvent varier d'un découpage
      à l'autre.
    - **Démonstrateur de régression** : seuls 5 curseurs sont proposés. Les autres
      variables (speechiness, instrumentalness, liveness, durée, tonalité, mode) sont
      fixées sur les valeurs d'un morceau de référence, ce qui limite la finesse des
      prédictions affichées.
    """
)

# =========================================================
# Pistes d'amélioration
# =========================================================
st.header("Pistes d'amélioration")
st.markdown(
    """
    - **Optimiser les modèles** : recherche d'hyperparamètres (`GridSearchCV`) pour la forêt
      aléatoire et le KNN, et essai d'autres algorithmes (gradient boosting, XGBoost).
    - **Fiabiliser l'évaluation** : validation croisée, et analyse de l'importance des
      variables pour comprendre ce qui pèse vraiment dans les prédictions.
    - **Traiter les confusions entre genres** : pondération des classes
      (`class_weight="balanced"`) ou regroupement des genres trop proches.
    - **Utiliser les sous-genres** (`playlist_subgenre`) pour obtenir des classes plus
      homogènes sur le plan audio.
    - **Enrichir les données** : jeu de données plus récent, paroles, popularité de
      l'artiste, ou nouvelles variables construites à partir des existantes.
    - **Compléter le démonstrateur de régression** : proposer tous les curseurs ou
      permettre de charger les attributs d'un morceau existant, comme dans la page
      Classification.
    """
)