# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(page_title="Dashboard Operacional ETE", layout="wide")

# =========================
# GOOGLE SHEETS
# =========================
SHEET_ID = "1Gv0jhdQLaGkzuzDXWNkD0GD5OMM84Q_zkOkQHGBhLjU"
GID = "1283870792"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

# Leitura e normalização de colunas
df = pd.read_csv(CSV_URL)
# Remove espaços e normaliza para minúsculas sem acentos para facilitar os filtros
def _strip_accents(s: str) -> str:
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

original_columns = list(df.columns)
df.columns = [str(c).strip() for c in df.columns]
cols_lower_noacc = [_strip_accents(c.lower()) for c in df.columns]

# Mapa auxiliar: coluna normalizada -> nome original (para buscar valores)
COLMAP = dict(zip(cols_lower_noacc, df.columns))

# Coluna de tempo (mantém nome original se existir; caso contrário, usa fallback)
TIME_CANDIDATES = [
    "carimbo de data/hora", "timestamp", "data", "hora", "date", "datetime"
]
TIME_COL = None
for c in df.columns:
    if _strip_accents(c.lower()) in [_strip_accents(x) for x in TIME_CANDIDATES]:
        TIME_COL = c
        break

# =========================
# DICIONÁRIOS / CONFIG
# =========================

# Renomeia caçambas (aplique aqui os aliases conforme necessário)
NOME_LIMPO = {
    "nivel da cacamba 1": "Caçamba 1",
    "nivel da cacamba 2": "Caçamba 2",
    "nivel da cacamba 3": "Caçamba 3",
    "nivel de cacamba 1": "Caçamba 1",
    "nivel de cacamba 2": "Caçamba 2",
    "nivel de cacamba 3": "Caçamba 3",
    "nivel da caçamba 1": "Caçamba 1",
    "nivel da caçamba 2": "Caçamba 2",
    "nivel da caçamba 3": "Caçamba 3",
    # Adicione variações que existirem na planilha:
    # "nivel cacamba nitr 1": "Caçamba Nitr 1",
    # "nivel cacamba mbbr 1": "Caçamba MBBR 1",
}

# Palavras-chave para split por área
KW_CACAMBA = ["cacamba", "caçamba"]
KW_NITR = ["nitr", "nitrificacao", "nitrificação"]
KW_MBBR = ["mbbr"]

# Palavras-chave para tiles
KW_VALVULA = ["valvula", "válvula"]
KW_SOPRADOR = ["soprador"] + KW_NITR + KW_MBBR

# =========================
# FUNÇÕES AUXILIARES
# =========================

def to_float_ptbr(x):
    """Converte string PT-BR (%, vírgula etc.) para float."""
    if pd.isna(x):
        return np.nan

    s = str(x).strip().replace("%", "")

    # Formatos comuns: "10,5" ou "1.234,5" ou "10.5"
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    elif "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")

    try:
        return float(s)
    except:
        return np.nan


def last_valid_raw(df, col):
    """Retorna o último valor não vazio da coluna (texto bruto)."""
    s = df[col].replace(r"^\s*$", np.nan, regex=True)
    valid = s.dropna()
    if valid.empty:
        return None
    return valid.iloc[-1]


def _filter_columns_by_keywords(all_cols_norm_noacc, keywords):
    """Retorna lista de colunas (nomes originais) que contém TODAS/QUALQUER keywords."""
    kws = [_strip_accents(k.lower()) for k in keywords]
    selected_norm = []
    for c_norm in all_cols_norm_noacc:
        if any(k in c_norm for k in kws):
            selected_norm.append(c_norm)
    # Retorna nos nomes originais mantendo a ordem de aparição
    return [COLMAP[c] for c in selected_norm]


def _nome_exibicao(label_original: str) -> str:
    """Aplica NOME_LIMPO considerando normalização."""
    base = _strip_accents(label_original.lower())
    return NOME_LIMPO.get(base, label_original)


# =========================
# VELOCÍMETRO
# =========================

def make_speedometer(val, label):
    nome_exibicao = _nome_exibicao(label)

    if val is None or np.isnan(val):
        val = 0

    color = "#43A047" if val >= 70 else "#FB8C00" if val >= 30 else "#E53935"

    return go.Indicator(
        mode="gauge+number",
        value=float(val),
        number={"suffix": "%"},
        title={"text": f"<b>{nome_exibicao}</b>", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
        },
        domain={"x": [0, 1], "y": [0, 1]},
    )


def render_dashboard(title, filter_words, sort_alpha=True, n_cols=4):
    """Cria grade de velocímetros (gauge) para colunas filtradas."""
    cols_orig = _filter_columns_by_keywords(cols_lower_noacc, filter_words)

    if sort_alpha:
        cols_orig = sorted(cols_orig, key=lambda x: _nome_exibicao(x))

    if not cols_orig:
        st.info(f"Nenhuma coluna encontrada para: {', '.join(filter_words)}")
        return

    n_rows = int(np.ceil(len(cols_orig) / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "indicator"}] * n_cols for _ in range(n_rows)],
        subplot_titles=None,
        horizontal_spacing=0.05,
        vertical_spacing=0.15
    )

    for i, c in enumerate(cols_orig):
        raw = last_valid_raw(df, c)
        val = to_float_ptbr(raw)

        r = i // n_cols + 1
        cc = i % n_cols + 1

        fig.add_trace(make_speedometer(val, c), row=r, col=cc)

    # Altura proporcional à quantidade de linhas
    fig.update_layout(
        height=max(280 * n_rows, 280),
        margin=dict(l=10, r=10, t=10, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================
# TILES (Válvulas / Sopradores)
# =========================

def render_tiles(title, filter_words, n_cols=4):
    cols_orig = _filter_columns_by_keywords(cols_lower_noacc, filter_words)
    cols_orig = sorted(cols_orig, key=lambda x: _nome_exibicao(x))

    if not cols_orig:
        st.info(f"Nenhuma coluna encontrada para: {', '.join(filter_words)}")
        return

    fig = go.Figure()
    n_rows = int(np.ceil(len(cols_orig) / n_cols))

    fig.update_xaxes(visible=False, range=[0, n_cols])
    fig.update_yaxes(visible=False, range=[0, n_rows])

    for i, c in enumerate(cols_orig):
        raw = last_valid_raw(df, c)
        val = to_float_ptbr(raw)

        # Define preenchimento e texto
        if raw is None:
            fill = "#9E9E9E"
            txt = "—"
        elif not np.isnan(val):
            fill = "#43A047" if val >= 70 else "#FB8C00" if val >= 30 else "#E53935"
            txt = f"{val:.1f}%"
        else:
            txt = str(raw)
            t = _strip_accents(txt.lower())
            if t in ["ok", "ligado", "aberto", "rodando", "on"]:
                fill = "#43A047"
            elif t in ["nok", "falha", "erro", "off"]:
                fill = "#E53935"
            else:
                fill = "#FB8C00"

        r = i // n_cols
        cc = i % n_cols

        x0, x1 = cc + 0.05, cc + 0.95
        y0, y1 = (n_rows - 1 - r) + 0.05, (n_rows - 1 - r) + 0.95

        # Retângulo do card
        fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=y0, y1=y1,
            fillcolor=fill,
            line=dict(color="white", width=1)
        )

        # Texto central (valor) + subtítulo (nome)
        nome = _nome_exibicao(c)
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2 + 0.15,
            text=f"<b style='font-size:18px'>{txt}</b>",
            showarrow=False,
            font=dict(color="white"),
        )
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2 - 0.15,
            text=f"<span style='font-size:12px'>{nome}</span>",
            showarrow=False,
            font=dict(color="white"),
        )

    fig.update_layout(
        height=max(170 * n_rows, 170),
        margin=dict(l=10, r=10, t=10, b=10),
    )

    st.subheader(title)
    st.plotly_chart(fig, use_container_width=True)


# =========================
# DASHBOARD
# =========================

st.title("Dashboard Operacional ETE")

# ---- Caçambas: blocos separados por processo
st.subheader("Caçambas – Nitrificação")
render_dashboard("Caçambas – Nitrificação", KW_CACAMBA + KW_NITR)

st.subheader("Caçambas – MBBR")
render_dashboard("Caçambas – MBBR", KW_CACAMBA + KW_MBBR)

# ---- Válvulas
render_tiles("Válvulas", KW_VALVULA)

# ---- Sopradores (inclui chaves de nitr e mbbr)
render_tiles("Sopradores", KW_SOPRADOR)
