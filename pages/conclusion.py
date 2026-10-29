import streamlit as st

st.title("🏁 Conclusion")

# =========================================================
# Réponse à la problématique
# =========================================================
st.header("Réponse à la problématique")
st.markdown(
    """
    **Rappel de la problématique :** reformule ici la question posée en page Accueil
    (par exemple : *peut-on prédire la popularité ou le genre d'un titre à partir de
    ses caractéristiques audio ?*).

    **Réponse :** résume en quelques phrases ce que vos analyses et vos modèles
    permettent de conclure, en t'appuyant sur des chiffres.
    """
)

# Résultats clés des modèles : remplace les "—" par vos vraies valeurs
st.subheader("Résultats clés")
tab_reg, tab_clf = st.tabs(["Régression", "Classification"])

with tab_reg:
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", "—")
    c2.metric("MAE", "—")
    c3.metric("RMSE", "—")
    st.markdown("**Commentaire :** le meilleur modèle de régression est ..., car ...")

with tab_clf:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", "—")
    c2.metric("Precision", "—")
    c3.metric("Recall", "—")
    c4.metric("F1-score", "—")
    st.markdown("**Commentaire :** le meilleur modèle de classification est ..., car ...")

st.divider()

# =========================================================
# Limites
# =========================================================
st.header("Limites des modèles")
st.markdown(
    """
    Exemples à adapter à vos résultats :
    - **Information limitée** : les caractéristiques audio seules n'expliquent qu'une
      partie de la popularité d'un titre (marketing, notoriété de l'artiste, playlists...).
    - **Étiquette de genre** : `playlist_genre` vient de la playlist, pas du titre
      lui-même, et les doublons ont été supprimés en gardant la première occurrence.
    - **Déséquilibre ou chevauchement des classes** : certains genres se ressemblent
      beaucoup sur le plan audio.
    - **Données figées** : le jeu de données date de 2020 et ne reflète pas les
      tendances actuelles.
    - **Évaluation** : un seul découpage train/test, pas de validation croisée.
    """
)

# =========================================================
# Pistes d'amélioration
# =========================================================
st.header("Pistes d'amélioration")
st.markdown(
    """
    Exemples à adapter :
    - tester d'autres modèles (gradient boosting, réseaux de neurones) et optimiser
      les hyperparamètres (`GridSearchCV`) ;
    - utiliser la validation croisée pour des scores plus fiables ;
    - créer de nouvelles variables (année de sortie, nombre de titres de l'artiste...) ;
    - enrichir les données avec d'autres sources (paroles, historique d'écoute) ;
    - corriger le déséquilibre des classes (pondération, rééchantillonnage).
    """
)