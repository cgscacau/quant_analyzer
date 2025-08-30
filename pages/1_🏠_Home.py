import streamlit as st
from core.ui import app_header, next_steps_card

st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")

app_header("🏠 Home", "Visão geral do projeto & trilha de implementação")

st.success("Este é o *esqueleto*. Nada aqui é recomendação; vamos preencher aos poucos.")
next_steps_card([
    "Definir lista de ativos padrão (BR/US/Cripto).",
    "Padronizar funções de dados (Yahoo Finance).",
    "Estabelecer tema visual e componentes de UI.",
])
