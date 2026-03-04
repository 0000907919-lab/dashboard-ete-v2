# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import unicodedata
import re

# ==========================================================
# CONFIGURAÇÃO
# ==========================================================
st.set_page_config(page_title="Dashboard Operacional ETE", layout="wide")

SHEET_ID = "1Gv0jhdQLaGkzuzDXWNkD0GD5OMM84Q_zkOkQHGBhLjU"
GID_FORM = "1283870792"

CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID_FORM}"
df = pd.read_csv(CSV_URL)
df.columns = [str(c).strip() for c in df.columns]

# ==========================================================
# UTILITÁRIOS
# ==========================================================
def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s))
                   if unicodedata.category(c) != "Mn")

def to_float_ptbr(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("%","").strip()
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    elif "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan

def last_valid(col):
    s = df[col].replace(r"^\s*$", np.nan, regex=True).dropna()
    return s.iloc[-1] if not s.empty else None

# ==========================================================
# CORES SEMÁFORO
# ==========================================================
VERDE = "#2E7D32"
LARANJA = "#F57C00"
VERMELHO = "#C62828"
CINZA = "#607D8B"

# ==========================================================
# 1️⃣  CAÇAMBAS = VELOCÍMETRO (SOMENTE ELAS)
# ==========================================================
def render_cacambas():

    cols = [c for c in df.columns
            if "cacamba" in strip_accents(c.lower())]

    if not cols:
        return

    st.subheader("Caçambas")

    fig = make_subplots(
        rows=1,
        cols=len(cols),
        specs=[[{"type":"indicator"}]*len(cols)]
    )

    for i, col in enumerate(cols):
        val = to_float_ptbr(last_valid(col))
        if np.isnan(val):
            val = 0

        cor = VERDE if val >= 70 else LARANJA if val >= 30 else VERMELHO

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=val,
                number={'suffix': "%"},
                title={'text': col},
                gauge={
                    'axis': {'range': [0,100]},
                    'bar': {'color': cor}
                }
            ),
            row=1, col=i+1
        )

    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 2️⃣  TODO O RESTO = RETÂNGULOS
# ==========================================================
def cor_semaforo(valor):

    if valor is None:
        return CINZA

    texto = strip_accents(str(valor).lower())

    if texto in ["ok","ligado","on","aberto"]:
        return VERDE
    if texto in ["nok","falha","erro","off","fechado"]:
        return VERMELHO

    num = to_float_ptbr(valor)
    if not np.isnan(num):
        return VERDE if num >= 70 else LARANJA if num >= 30 else VERMELHO

    return CINZA


def render_tiles():

    # REMOVE CAÇAMBAS
    cols = [c for c in df.columns
            if "cacamba" not in strip_accents(c.lower())]

    if not cols:
        return

    st.subheader("Indicadores Operacionais")

    n_cols = 4
    fig = go.Figure()

    n_rows = int(np.ceil(len(cols)/n_cols))

    fig.update_xaxes(visible=False, range=[0,n_cols])
    fig.update_yaxes(visible=False, range=[0,n_rows])

    for i, col in enumerate(cols):

        raw = last_valid(col)
        cor = cor_semaforo(raw)

        r = i // n_cols
        c = i % n_cols

        x0,x1 = c+0.05, c+0.95
        y0,y1 = (n_rows-1-r)+0.05, (n_rows-1-r)+0.95

        fig.add_shape(
            type="rect",
            x0=x0,x1=x1,y0=y0,y1=y1,
            fillcolor=cor,
            line=dict(color="white")
        )

        fig.add_annotation(
            x=(x0+x1)/2,
            y=(y0+y1)/2,
            text=f"<b>{raw}</b><br><span style='font-size:11px'>{col}</span>",
            showarrow=False,
            font=dict(color="white")
        )

    fig.update_layout(
        height=max(180*n_rows,300),
        margin=dict(l=10,r=10,t=10,b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# DASHBOARD
# ==========================================================
st.title("Dashboard Operacional ETE")

render_cacambas()   # 🔵 SOMENTE CAÇAMBAS VIRAM VELOCÍMETRO
render_tiles()      # 🟩 TODO RESTO VIRA RETÂNGULO
