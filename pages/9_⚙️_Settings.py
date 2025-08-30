import streamlit as st
from core.ui import app_header

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
app_header("⚙️ Settings", "Preferências gerais (placeholder)")

st.toggle("Tema escuro (visual)", value=True, disabled=True)
st.text_input("Pasta de dados (local)", value="./data", disabled=True)
st.caption("Vamos habilitar ajustes reais aqui conforme avançarmos.")
