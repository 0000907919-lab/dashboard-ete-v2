import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

st.set_page_config(layout="wide")

# =========================
# GOOGLE SHEETS
# =========================

SHEET_ID = "1Gv0jhdQLaGkzuzDXWNkD0GD5OMM84Q_zkOkQHGBhLjU"
GID = "1283870792"

CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

df = pd.read_csv(CSV_URL)
df.columns = [str(c).strip() for c in df.columns]

TIME_COL = "Carimbo de data/hora"

# =========================
# FUNÇÕES AUXILIARES
# =========================

def to_float_ptbr(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip().replace("%", "")

    if "," in s and "." not in s:
        s = s.replace(",", ".")
    elif "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")

    try:
        return float(s)
    except:
        return np.nan


def last_valid_raw(df, col):
    s = df[col].replace(r"^\s*$", np.nan, regex=True)
    valid = s.dropna()

    if valid.empty:
        return None

    return valid.iloc[-1]


# =========================
# VELOCÍMETRO
# =========================

def make_speedometer(val, label):
    if val is None or np.isnan(val):
        val = 0

    color = "#43A047" if val >= 70 else "#FB8C00" if val >= 30 else "#E53935"

    return go.Indicator(
        mode="gauge+number",
        value=float(val),
        number={"suffix": "%"},
        title={"text": label},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
        }
    )


def render_dashboard(title, filter_words):
    cols = [c for c in df.columns if any(w in c.lower() for w in filter_words)]

    n_cols = 4
    n_rows = int(np.ceil(len(cols) / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "indicator"}] * n_cols for _ in range(n_rows)]
    )

    for i, c in enumerate(cols):
        raw = last_valid_raw(df, c)
        val = to_float_ptbr(raw)

        r = i // n_cols + 1
        col = i % n_cols + 1

        fig.add_trace(make_speedometer(val, c), row=r, col=col)

    fig.update_layout(height=250 * n_rows)

    st.plotly_chart(fig, use_container_width=True)


# =========================
# TILES (Válvulas / Sopradores)
# =========================

def render_tiles(title, filter_words):
    cols = [c for c in df.columns if any(w in c.lower() for w in filter_words)]

    fig = go.Figure()
    n_cols = 4
    n_rows = int(np.ceil(len(cols) / n_cols))

    fig.update_xaxes(visible=False, range=[0, n_cols])
    fig.update_yaxes(visible=False, range=[0, n_rows])

    for i, c in enumerate(cols):
        raw = last_valid_raw(df, c)
        val = to_float_ptbr(raw)

        if raw is None:
            fill = "#9E9E9E"
            txt = "—"
        elif not np.isnan(val):
            fill = "#43A047" if val >= 70 else "#FB8C00" if val >= 30 else "#E53935"
            txt = f"{val:.1f}%"
        else:
            txt = str(raw)
            t = txt.lower()
            if t in ["ok", "ligado", "aberto", "rodando"]:
                fill = "#43A047"
            elif t in ["nok", "falha", "erro"]:
                fill = "#E53935"
            else:
                fill = "#FB8C00"

        r = i // n_cols
        cc = i % n_cols

        x0, x1 = cc + 0.05, cc + 0.95
        y0, y1 = (n_rows - 1 - r) + 0.05, (n_rows - 1 - r) + 0.95

        fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=y0, y1=y1,
            fillcolor=fill,
            line=dict(color="white")
        )

        fig.add_annotation(
            x=(x0+x1)/2,
            y=(y0+y1)/2,
            text=f"<b>{txt}</b><br><span style='font-size:11px'>{c}</span>",
            showarrow=False,
            font=dict(color="white")
        )

    fig.update_layout(height=170 * n_rows)

    st.plotly_chart(fig, use_container_width=True)


# =========================
# DASHBOARD
# =========================

st.title("Dashboard Operacional ETE")

render_dashboard("Caçambas", ["caçamba", "cacamba"])

render_tiles("Válvulas", ["válvula", "valvula"])

render_tiles("Sopradores", ["soprador", "mbbr", "nitr"])
