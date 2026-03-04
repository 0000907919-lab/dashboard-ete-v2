# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import io, requests, re
from matplotlib.ticker import FuncFormatter

# =========================

# CONFIGURAÇÃO DA PÁGINA

# =========================

st.set_page_config(page_title="Dashboard Operacional ETE", layout="wide")

# =========================

# GOOGLE SHEETS – ABA 1

# =========================

SHEET_ID = "1Gv0jhdQLaGkzuzDXWNkD0GD5OMM84Q_zkOkQHGBhLjU"
GID_FORM = "1283870792"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID_FORM}"

# =========================

# CARREGAMENTO + LIMPEZA

# =========================

df = pd.read_csv(CSV_URL)
df.columns = [str(c).strip() for c in df.columns]

# ---- REMOVER COLUNAS DUPLICADAS ----

df = df.loc[:, ~df.columns.duplicated()]

# ---- REMOVER LINHAS DUPLICADAS (mantém última ocorrência) ----

df = df.drop_duplicates(keep="last").reset_index(drop=True)

# ---- SE EXISTIR CARIMBO DE DATA, MANTER REGISTRO MAIS RECENTE POR DATA ----

for col in ["Carimbo de data/hora", "Data"]:
if col in df.columns:
df[col] = pd.to_datetime(df[col], errors="coerce")
df = df.sort_values(col)
df = df.drop_duplicates(subset=[col], keep="last")
df = df.reset_index(drop=True)

# ============================================================

# A PARTIR DAQUI O RESTANTE DO SEU CÓDIGO PERMANECE IGUAL

# (todas as funções, gauges, tiles, cartas, etc.)

# ============================================================

# =========================

# CARTAS DE CONTROLE — CUSTOS

# =========================

st.markdown("---")
st.header("🔴 Cartas de Controle — Custo (R$)")

with st.sidebar:
gid_input = st.text_input("GID da aba de gastos", value="668859455")

CC_GID_GASTOS = gid_input.strip() or "668859455"
CC_URL_GASTOS = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={CC_GID_GASTOS}"

@st.cache_data(ttl=900)
def cc_baixar_csv_bruto(url: str):
resp = requests.get(url, timeout=20)
resp.raise_for_status()
buf = io.StringIO(resp.text)
df_txt = pd.read_csv(buf, dtype=str, keep_default_na=False, header=None)
return df_txt

try:
cc_df_raw = cc_baixar_csv_bruto(CC_URL_GASTOS)
except Exception as e:
st.error(f"Erro ao carregar aba de custos: {e}")
st.stop()

# ---- Detecta cabeçalho ----

def cc_strip_acc_lower(s):
import unicodedata
s = unicodedata.normalize("NFKD", str(s))
s = "".join(ch for ch in s if not unicodedata.combining(ch))
return s.lower().strip()

def cc_find_header_row(df_txt, max_scan=120):
for i in range(min(len(df_txt), max_scan)):
row_vals = [cc_strip_acc_lower(x) for x in df_txt.iloc[i].tolist()]
if any("data" in v for v in row_vals) and any("custo" in v or "valor" in v for v in row_vals):
return i
return None

cc_hdr = cc_find_header_row(cc_df_raw)

if cc_hdr is None:
st.error("Cabeçalho de custos não encontrado.")
st.stop()

cc_header_vals = [str(x).strip() for x in cc_df_raw.iloc[cc_hdr].tolist()]
cc_df_all = cc_df_raw.iloc[cc_hdr + 1:].copy()
cc_df_all.columns = cc_header_vals

# ---- REMOVER COLUNAS DUPLICADAS ----

cc_df_all = cc_df_all.loc[:, ~cc_df_all.columns.duplicated()]

# ---- REMOVER LINHAS DUPLICADAS (mantém última) ----

cc_df_all = cc_df_all.drop_duplicates(keep="last").reset_index(drop=True)

# ============================================================

# RESTANTE DAS CARTAS CONTINUA IGUAL

# (agrupamentos diário, semanal, mensal e gráficos)

# ============================================================

st.success("✅ Duplicatas removidas automaticamente com sucesso.")
