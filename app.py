import streamlit as st

st.set_page_config(page_title="Projet Spotify", page_icon="🎵", layout="wide")

pg = st.navigation([
    st.Page("pages/accueil.py", title="Accueil", icon="🏠", default=True),
    st.Page("pages/exploration.py", title="Exploration des données", icon="📊"),
    st.Page("pages/regression.py", title="Régression", icon="📈"),
    st.Page("pages/classification.py", title="Classification", icon="🎯"),
    st.Page("pages/conclusion.py", title="Conclusion", icon="🏁"),
])
pg.run()