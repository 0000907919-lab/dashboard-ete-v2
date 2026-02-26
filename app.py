# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# -------------------------
# Carrega a planilha
# -------------------------
df = pd.read_csv(CSV_URL)

# Remove espaços e normaliza nomes de colunas
df.columns = [str(c).strip() for c in df.columns]

# Função para remover acentos
def _strip_accents(s: str) -> str:
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

# Tabela auxiliar com nomes normalizados (p/ filtros)
cols_lower_noacc = [_strip_accents(c.lower()) for c in df.columns]
COLMAP = dict(zip(cols_lower_noacc, df.columns))  # normalizado -> original

# (Opcional) detectar coluna de tempo
TIME_CANDIDATES = ["carimbo de data/hora", "timestamp", "data", "hora", "date", "datetime"]
TIME_COL = None
for c in df.columns:
    if _strip_accents(c.lower()) in [_strip_accents(x) for x in TIME_CANDIDATES]:
        TIME_COL = c
        break

# =========================
# PALAVRAS-CHAVE E NOMES
# =========================
KW_CACAMBA = ["cacamba", "caçamba"]
KW_NITR = ["nitr", "nitrificacao", "nitrificação"]
KW_MBBR = ["mbbr"]
KW_VALVULA = ["valvula", "válvula"]
KW_SOPRADOR = ["soprador", "oxigenacao", "oxigenação"]

# Dicionário para renomear Caçambas (adicione variações que aparecerem na planilha)
NOME_LIMPO = {
    "nivel da cacamba 1": "Caçamba 1",
    "nivel da cacamba 2": "Caçamba 2",
    "nivel da cacamba 3": "Caçamba 3",
    "nivel cacamba 1": "Caçamba 1",
    "nivel cacamba 2": "Caçamba 2",
    "nivel cacamba 3": "Caçamba 3",
    "nivel de cacamba 1": "Caçamba 1",
    "nivel de cacamba 2": "Caçamba 2",
    "nivel de cacamba 3": "Caçamba 3",
    # Exemplos (descomente/edite se tiver na sua planilha):
    # "nivel cacamba nitr 1": "Caçamba Nitr 1",
    # "nivel cacamba mbbr 1": "Caçamba MBBR 1",
}

# =========================
# FUNÇÕES AUXILIARES
# =========================
def to_float_ptbr(x):
    """Converte string PT-BR (%, vírgula) para float."""
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
    """Último valor não vazio de uma coluna."""
    s = df[col].replace(r"^\s*$", np.nan, regex=True)
    valid = s.dropna()
    if valid.empty:
        return None
    return valid.iloc[-1]

def _filter_columns_by_keywords(all_cols_norm_noacc, keywords):
    """Retorna nomes originais das colunas que contenham QUALQUER keyword."""
    kws = [_strip_accents(k.lower()) for k in keywords]
    selected_norm = []
    for c_norm in all_cols_norm_noacc:
        if any(k in c_norm for k in kws):
            selected_norm.append(c_norm)
    return [COLMAP[c] for c in selected_norm]

def _nome_exibicao(label_original: str) -> str:
    base = _strip_accents(label_original.lower())
    return NOME_LIMPO.get(base, label_original)

# =========================
# GAUGES (somente Caçambas)
# =========================
def make_speedometer(val, label):
    nome_exibicao = _nome_exibicao(label)
    if val is None or np.isnan(val):
        val = 0.0

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

def render_cacambas_gauges(title, n_cols=4):
    """Renderiza TODAS as caçambas como velocímetro (sem dividir por processo)."""
    # Procura por colunas que contenham 'caçamba'
    cols_orig = _filter_columns_by_keywords(cols_lower_noacc, KW_CACAMBA)
    # Mantém apenas as que realmente são de nível (evita pegar válvulas/sopradores)
    cols_orig = [c for c in cols_orig if any(k in _strip_accents(c.lower()) for k in KW_CACAMBA)]
    cols_orig = sorted(cols_orig, key=lambda x: _nome_exibicao(x))

    if not cols_orig:
        st.info("Nenhuma caçamba encontrada.")
        return

    n_rows = int(np.ceil(len(cols_orig) / n_cols))
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "indicator"}] * n_cols for _ in range(n_rows)],
        horizontal_spacing=0.05,
        vertical_spacing=0.15
    )

    for i, c in enumerate(cols_orig):
        raw = last_valid_raw(df, c)
        val = to_float_ptbr(raw)
        r = i // n_cols + 1
        cc = i % n_cols + 1
        fig.add_trace(make_speedometer(val, c), row=r, col=cc)

    fig.update_layout(
        height=max(280 * n_rows, 280),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.subheader(title)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TILES (Válvulas / Sopradores)
# =========================
def _render_tiles_from_cols(title, cols_orig, n_cols=4):
    cols_orig = sorted(cols_orig, key=lambda x: _nome_exibicao(x))
    if not cols_orig:
        st.info(f"Nenhum item encontrado para: {title}")
        return

    fig = go.Figure()
    n_rows = int(np.ceil(len(cols_orig) / n_cols))

    fig.update_xaxes(visible=False, range=[0, n_cols])
    fig.update_yaxes(visible=False, range=[0, n_rows])

    for i, c in enumerate(cols_orig):
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
            t = _strip_accents(txt.lower())
            if t in ["ok", "ligado", "aberto", "rodando", "on"]:
                fill = "#43A047"
            elif t in ["nok", "falha", "erro", "fechado", "off"]:
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
            line=dict(color="white", width=1)
        )

        nome = _nome_exibicao(c)
        # Valor
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2 + 0.15,
            text=f"<b style='font-size:18px'>{txt}</b>",
            showarrow=False,
            font=dict(color="white"),
        )
        # Nome do item
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

def render_tiles_split(title_base, base_keywords, n_cols=4):
    """
    Renderiza duas seções de cards: Nitrificação e MBBR,
    para os grupos especificados por base_keywords (ex.: válvulas ou sopradores).
    """
    # Nitrificação
    cols_nitr = _filter_columns_by_keywords(cols_lower_noacc, base_keywords + KW_NITR)
    # Remove quaisquer colunas que sejam caçamba (só por garantia)
    cols_nitr = [c for c in cols_nitr if not any(k in _strip_accents(c.lower()) for k in KW_CACAMBA)]
    _render_tiles_from_cols(f"{title_base} – Nitrificação", cols_nitr, n_cols=n_cols)

    # MBBR
    cols_mbbr = _filter_columns_by_keywords(cols_lower_noacc, base_keywords + KW_MBBR)
    cols_mbbr = [c for c in cols_mbbr if not any(k in _strip_accents(c.lower()) for k in KW_CACAMBA)]
    _render_tiles_from_cols(f"{title_base} – MBBR", cols_mbbr, n_cols=n_cols)

# =========================
# DASHBOARD
# =========================
st.title("Dashboard Operacional ETE")

# ---- Caçambas (somente velocímetro, todas juntas)
render_cacambas_gauges("Caçambas")

# ---- Válvulas (cards) — dividido em Nitrificação e MBBR
render_tiles_split("Válvulas", KW_VALVULA)

# ---- Sopradores (cards) — dividido em Nitrificação e MBBR
render_tiles_split("Sopradores", KW_SOPRADOR)def _nome_exibicao(label_original: str) -> str:
    """Padroniza nomes de Caçambas, Sopradores e Válvulas."""

    base = _strip_accents(label_original.lower()).strip()

    # -------------------------
    # 1) CAÇAMBAS
    # -------------------------
    if "cacamba" in base:
        # extrai número
        num = ''.join(filter(str.isdigit, base))
        if num:
            return f"Nível da caçamba {num}"
        return "Nível da caçamba"

    # -------------------------
    # 2) SOPRADORES (Nitrificação / MBBR)
    # -------------------------
    if any(k in base for k in ["soprador", "oxigenacao", "oxigenacao", "oxigenacao"]):

        num = ''.join(filter(str.isdigit, base))
        
        if "nitr" in base:
            return f"Soprador de nitrificação {num}" if num else "Soprador de nitrificação"

        if "mbbr" in base:
            return f"Soprador de MBBR {num}" if num else "Soprador de MBBR"

        return f"Soprador {num}" if num else "Soprador"

    # -------------------------
    # 3) VÁLVULAS
    # -------------------------
    if "valvula" in base:

        num = ''.join(filter(str.isdigit, base))

        if "nitr" in base:
            return f"Válvula de nitrificação {num}" if num else "Válvula de nitrificação"

        if "mbbr" in base:
            return f"Válvula de MBBR {num}" if num else "Válvula de MBBR"

        return f"Válvula {num}" if num else "Válvula"

    # fallback
    return label_original
