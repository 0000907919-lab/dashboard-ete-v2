# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import unicodedata

st.set_page_config(page_title="Dashboard Operacional ETE", layout="wide")
SHEET_ID = "1Gv0jhdQLaGkzuzDXWNkD0GD5OMM84Q_zkOkQHGBhLjU"
GID_FORM = "1283870792"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID_FORM}"

@st.cache_data(ttl=300)
def load_df():
    df = pd.read_csv(CSV_URL, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_df()

# Utilities
def strip_acc(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def to_float(x):
    if x is None: return np.nan
    s = str(x).replace('%','').strip()
    if "," in s and "." not in s: s=s.replace(",",".")
    elif "." in s and "," in s: s=s.replace(".","").replace(",",".")
    try: return float(s)
    except: return np.nan

cols_norm=[strip_acc(c.lower()) for c in df.columns]
COLMAP=dict(zip(cols_norm,df.columns))

KW_CAC=["cacamba","caçamba"]
KW_OX=["oxigenacao","oxigenação"]
KW_SOP=["soprador"]
KW_VAZ=["vazao","vazão"]
KW_PH=["ph"]
KW_SST=["sst","ss "]
KW_DQO=["dqo"]
KW_NITR=["nitr"]
KW_MBBR=["mbbr"]

# Header
def header():
    st.title("Dashboard Operacional ETE")
    if 'carimbo de data/hora' in [c.lower() for c in df.columns]:
        col= [c for c in df.columns if c.lower()=="carimbo de data/hora"][0]
        st.metric("Último registro", df[col].dropna().iloc[-1])
    st.metric("Registros", len(df))

header()

# Gauges only for Caçambas
def nome(label):
    base=strip_acc(label.lower())
    nums="".join(ch for ch in label if ch.isdigit())
    if "cacamba" in base: return f"Nível da Caçamba {nums}" if nums else "Nível da Caçamba"
    return label

st.subheader("Caçambas")
cols_cac=[COLMAP[c] for c in cols_norm if any(k in c for k in KW_CAC)]
if not cols_cac:
    st.info("Nenhuma caçamba encontrada.")
else:
    fig=make_subplots(rows=1,cols=len(cols_cac),specs=[[{"type":"indicator"}]*len(cols_cac)])
    for i,col in enumerate(cols_cac):
        val=to_float(df[col].dropna().iloc[-1])
        color="#43A047" if val>=70 else "#FB8C00" if val>=30 else "#E53935"
        fig.add_trace(go.Indicator(mode="gauge+number",value=val,number={"suffix":"%"},title={"text":nome(col)},gauge={"axis":{"range":[0,100]},"bar":{"color":color}}),1,i+1)
    st.plotly_chart(fig,use_container_width=True)

# Tiles generic
def tiles(title, kws):
    cols=[COLMAP[c] for c in cols_norm if any(k in c for k in kws) and not any(x in c for x in KW_CAC)]
    if not cols: return
    st.subheader(title)
    fig=go.Figure()
    for i,col in enumerate(cols):
        raw=df[col].dropna().iloc[-1]
        val=to_float(raw)
        color="#546E7A"
        if val is not None and not np.isnan(val):
            color="#43A047" if val>=70 else "#FB8C00" if val>=30 else "#E53935"
        fig.add_shape(type="rect",x0=0,x1=1,y0=i,y1=i+0.9,fillcolor=color,line=dict(color="white"))
        fig.add_annotation(x=0.5,y=i+0.45,text=f"<b>{raw}</b><br>{nome(col)}",showarrow=False,font=dict(color="white"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=60*len(cols))
    st.plotly_chart(fig,use_container_width=True)

tiles("Oxigenação",KW_OX)
tiles("Sopradores",KW_SOP)
tiles("Vazões",KW_VAZ)
tiles("pH",KW_PH)
tiles("SST / SS",KW_SST)
tiles("DQO",KW_DQO)

st.markdown("---")
st.caption("Painel carregado com sucesso.")
