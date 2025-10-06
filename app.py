import streamlit as st

st.set_page_config(page_title="Analizador de Créditos", layout="wide")

st.title("Analizador de Créditos – Demo 🚀")
st.write("Si ves esto online, el deploy quedó **ok**.")

# Esqueleto (sin lógica aún)
base1 = st.file_uploader("Sube Base 1 (Aprobados) .xlsx", type=["xlsx"])
base2 = st.file_uploader("Sube Base 2 (Desembolsados) .csv", type=["csv"])

if base1 or base2:
    st.success("Archivos cargados (demo). La lógica vendrá después.")
