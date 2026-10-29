import pandas as pd
import streamlit as st
 
DATA_PATH = "data/spotify_clean.csv"
 
 
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)