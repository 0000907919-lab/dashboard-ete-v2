# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import re

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(page_title="Dashboard Operacional ETE", layout="wide")

# =========================
# GOOGLE SHEETS – ABA 1 (Operacional / Formulário)
# =========================
SHEET_ID = "1Gv0jhdQLaGkzuzDXWNkD0GD5OMM84Q_zkOkQHGBhLjU"
GID_FORM = "1283870792"
CSV_URL_FORM = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID_FORM}"

# Carrega a planilha operacional
df = pd.read_csv(CSV_URL_FORM)
df.columns = [str(c).strip() for c in df.columns]

# =========================
# PARÂMETROS DE REGRA DE STATUS
# =========================
# Tudo que for numérico em VÁLVULAS e SOPRADORES:
#   > limiar -> OK ; <= limiar -> OFF
BLOWER_O2_OK_THRESHOLD = 0.0     # ex.: 0.2 para exigir > 0,2
VALVE_NUMERIC_OK_THRESHOLD = 0.0 # ex.: 0.0 (0=OFF, >0=OK)

# =========================
# NORMALIZAÇÃO / AUXILIARES
# =========================
def _strip_accents(s: str) -> str:
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

def _slug(s: str) -> str:
    return _strip_accents(str(s).lower()).replace(" ", "-").replace("–", "-").replace("/", "-")

cols_lower_noacc = [_strip_accents(c.lower()) for c in df.columns]
COLMAP = dict(zip(cols_lower_noacc, df.columns))  # normalizado -> original

# Palavras‑chave
KW_CACAMBA = ["cacamba", "caçamba"]
KW_NITR = ["nitr", "nitrificacao", "nitrificação"]
KW_MBBR = ["mbbr"]
KW_VALVULA = ["valvula", "válvula"]
KW_SOPRADOR = ["soprador", "sopradores", "oxigenacao", "oxigenação"]

# Grupos adicionais
KW_NIVEIS_OUTROS = ["nivel", "nível"]  # exclui caçambas
KW_VAZAO = ["vazao", "vazão"]
KW_PH = ["ph ", " ph"]                 # evita bater em oxiPH
KW_SST = ["sst ", " sst", "ss "]
KW_DQO = ["dqo ", " dqo"]
KW_ESTADOS = ["tridecanter", "desvio", "tempo de descarte", "volante"]

# -------------------------
# Conversões e utilidades
# -------------------------
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

def last_valid_raw(df_local, col):
    """Último valor não vazio."""
    s = pd.Series(df_local[col])
    if pd.api.types.is_numeric_dtype(s):
        s = s.dropna()
        return None if s.empty else s.iloc[-1]
    s = s.replace(r"^\s*$", np.nan, regex=True).dropna()
    return None if s.empty else s.iloc[-1]

def _filter_columns_by_keywords(all_cols_norm_noacc, keywords):
    """Nomes originais de colunas que contenham QUALQUER keyword."""
    kws = [_strip_accents(k.lower()) for k in keywords]
    selected_norm = []
    for c_norm in all_cols_norm_noacc:
        if any(k in c_norm for k in kws):
            selected_norm.append(c_norm)
    return [COLMAP[c] for c in selected_norm]

def _extract_number_text(text: str) -> str:
    m = re.search(r'(\d+)', text or "")
    return m.group(1) if m else ""

def _extract_number_int(text: str) -> int:
    n = _extract_number_text(text)
    return int(n) if n else 9999

def _remove_brackets(text: str) -> str:
    return text.split("[", 1)[0].strip()

def _units_from_label(label: str) -> str:
    s = _strip_accents(label.lower())
    if "m3/h" in s or "m³/h" in label.lower():
        return " m³/h"
    if "mg/l" in s:
        return " mg/L"
    if "(%)" in label or "%" in label:
        return "%"
    return ""

# =========================
# PADRONIZAÇÃO DE NOMES (TÍTULOS)
# =========================
def re_replace_case_insensitive(s, pattern, repl):
    import re as _re
    return _re.sub(pattern, repl, s, flags=_re.IGNORECASE)

def _nome_exibicao(label_original: str) -> str:
    base_clean = _remove_brackets(label_original)
    base = _strip_accents(base_clean.lower()).strip()
    num = _extract_number_text(base)

    if "cacamba" in base:
        return f"Nível da caçamba {num}" if num else "Nível da caçamba"

    if ("soprador" in base) or ("oxigenacao" in base):
        if any(k in base for k in KW_NITR):
            return f"Soprador de nitrificação {num}" if num else "Soprador de nitrificação"
        if any(k in base for k in KW_MBBR):
            return f"Soprador de MBBR {num}" if num else "Soprador de MBBR"
        return f"Soprador {num}" if num else "Soprador"

    if "valvula" in base:
        if any(k in base for k in KW_NITR):
            return f"Válvula de nitrificação {num}" if num else "Válvula de nitrificação"
        if any(k in base for k in KW_MBBR):
            return f"Válvula de MBBR {num}" if num else "Válvula de MBBR"
        return f"Válvula {num}" if num else "Válvula"

    txt = base_clean
    replacements = {
        "ph": "pH", "dqo": "DQO", "sst": "SST", "ss ": "SS ",
        "vazao": "Vazão", "nível": "Nível", "nivel": "Nível",
        "mix": "MIX", "tq": "TQ", "mbbr": "MBBR",
        "nitrificacao": "Nitrificação", "nitrificação": "Nitrificação",
        "mab": "MAB",
    }
    for k, v in replacements.items():
        txt = re_replace_case_insensitive(txt, k, v)
    return txt.strip()

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
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}},
        domain={"x": [0, 1], "y": [0, 1]},
    )

def render_cacambas_gauges(title, n_cols=4):
    cols_orig = _filter_columns_by_keywords(cols_lower_noacc, KW_CACAMBA)
    cols_orig = [c for c in cols_orig if any(k in _strip_accents(c.lower()) for k in KW_CACAMBA)]
    cols_orig = sorted(cols_orig, key=lambda x: _nome_exibicao(x))
    if not cols_orig:
        st.info("Nenhuma caçamba encontrada.")
        return

    n_rows = int(np.ceil(len(cols_orig) / n_cols))
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=[[{"type": "indicator"}] * n_cols for _ in range(n_rows)],
        horizontal_spacing=0.05, vertical_spacing=0.15
    )
    for i, c in enumerate(cols_orig):
        raw = last_valid_raw(df, c)
        val = to_float_ptbr(raw)
        r = i // n_cols + 1
        cc = i % n_cols + 1
        fig.add_trace(make_speedometer(val, c), row=r, col=cc)

    fig.update_layout(height=max(280 * n_rows, 280), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True, key=f"plot-gauges-{_slug(title)}")

# =========================
# TILES (cards) – com regra numérica=OK p/ Válvulas & Sopradores
# =========================
def _status_from_raw_for_group(raw, group_type: str):
    """
    Regra: numérico => OK/ OFF por limiar; texto => OK/NOK/OFF.
    group_type: 'soprador'|'valvula'|'outros'
    """
    if raw is None:
        return "—", None

    s = str(raw).strip()
    v = to_float_ptbr(s)
    if not np.isnan(v):
        if group_type == "soprador":
            return ("OK" if v > BLOWER_O2_OK_THRESHOLD else "OFF"), v
        if group_type == "valvula":
            return ("OK" if v > VALVE_NUMERIC_OK_THRESHOLD else "OFF"), v
        # outros numéricos mantêm exibição numérica
        return None, v

    # texto
    t = _strip_accents(s.lower())
    if t in ["ok", "on", "ligado", "rodando", "aberto"]:
        return "OK", None
    if t in ["nok", "falha", "erro"]:
        return "NOK", None
    if t in ["off", "desligado", "fechado", "parado"]:
        return "OFF", None
    return s.upper(), None

def _tile_color_and_text(raw_value, label, interpret_numeric_as_status: bool):
    """Retorna (fill, txt) conforme a regra solicitada."""
    if interpret_numeric_as_status:
        # Distingue grupo pela label
        base = _strip_accents(label.lower())
        group_type = "soprador" if ("soprador" in base or "oxigenacao" in base) else ("valvula" if "valvula" in base else "outros")
        stt, v = _status_from_raw_for_group(raw_value, group_type)

        if stt in ["OK", "OFF", "NOK"]:
            fill = {"OK": "#43A047", "OFF": "#E53935", "NOK": "#E53935"}[stt]
            return fill, stt
        elif stt == "—":
            return "#9E9E9E", "—"
        elif stt is not None:  # outro texto
            return "#FB8C00", stt
        else:
            # numérico mas sem status (não deve acontecer aqui)
            v = v if v is not None else np.nan
            if np.isnan(v):
                return "#9E9E9E", "—"
            # fallback: verde se >0
            return ("#43A047" if v > 0 else "#E53935"), f"{v:.2f}"
    else:
        # Comportamento antigo para os demais grupos
        v = to_float_ptbr(raw_value)
        if raw_value is None:
            return "#9E9E9E", "—"
        if not np.isnan(v):
            units = _units_from_label(label)
            if units == "%":
                fill = "#43A047" if v >= 70 else "#FB8C00" if v >= 30 else "#E53935"
                return fill, f"{v:.1f}%"
            # neutro numérico
            return "#546E7A", f"{v:.2f}{units}"
        # texto
        t = _strip_accents(str(raw_value).strip().lower())
        if t in ["ok", "ligado", "aberto", "rodando", "on"]:
            return "#43A047", "OK"
        if t in ["nok", "falha", "erro", "fechado", "off"]:
            return "#E53935", t.upper()
        return "#FB8C00", str(raw_value)

def _render_tiles_from_cols(title, cols_orig, n_cols=4, interpret_numeric_as_status=False):
    cols_orig = [c for c in cols_orig if c]
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
        fill, txt = _tile_color_and_text(raw, c, interpret_numeric_as_status)

        r = i // n_cols
        cc = i % n_cols
        x0, x1 = cc + 0.05, cc + 0.95
        y0, y1 = (n_rows - 1 - r) + 0.05, (n_rows - 1 - r) + 0.95

        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=fill, line=dict(color="white", width=1))
        nome = _nome_exibicao(c)
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2 + 0.12,
                           text=f"<b style='font-size:18px'>{txt}</b>",
                           showarrow=False, font=dict(color="white"))
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2 - 0.18,
                           text=f"<span style='font-size:12px'>{nome}</span>",
                           showarrow=False, font=dict(color="white"))

    fig.update_layout(height=max(170 * n_rows, 170), margin=dict(l=10, r=10, t=10, b=10))
    st.subheader(title)
    st.plotly_chart(fig, use_container_width=True, key=f"plot-tiles-{_slug(title)}")

def render_tiles_split_status(title_base, base_keywords, n_cols=4):
    """Cards para Nitrificação e MBBR com regra numérica=OK."""
    # Nitrificação
    cols_nitr = _filter_columns_by_keywords(cols_lower_noacc, base_keywords + KW_NITR)
    cols_nitr = [c for c in cols_nitr if not any(k in _strip_accents(c.lower()) for k in KW_CACAMBA)]
    _render_tiles_from_cols(f"{title_base} – Nitrificação", cols_nitr, n_cols=n_cols, interpret_numeric_as_status=True)

    # MBBR
    cols_mbbr = _filter_columns_by_keywords(cols_lower_noacc, base_keywords + KW_MBBR)
    cols_mbbr = [c for c in cols_mbbr if not any(k in _strip_accents(c.lower()) for k in KW_CACAMBA)]
    _render_tiles_from_cols(f"{title_base} – MBBR", cols_mbbr, n_cols=n_cols, interpret_numeric_as_status=True)

# -------------------------
# Grupos adicionais (sem a regra numérica=OK)
# -------------------------
def render_outros_niveis():
    cols = _filter_columns_by_keywords(cols_lower_noacc, KW_NIVEIS_OUTROS)
    cols = [c for c in cols if not any(k in _strip_accents(c.lower()) for k in KW_CACAMBA)]
    if not cols:
        return
    _render_tiles_from_cols("Níveis (MAB/TQ de Lodo)", cols, n_cols=3, interpret_numeric_as_status=False)

def render_vazoes():
    cols = _filter_columns_by_keywords(cols_lower_noacc, KW_VAZAO)
    if not cols:
        return
    _render_tiles_from_cols("Vazões", cols, n_cols=3, interpret_numeric_as_status=False)

def render_ph():
    cols = _filter_columns_by_keywords(cols_lower_noacc, KW_PH)
    if not cols:
        return
    _render_tiles_from_cols("pH", cols, n_cols=4, interpret_numeric_as_status=False)

def render_sst():
    cols = _filter_columns_by_keywords(cols_lower_noacc, KW_SST)
    if not cols:
        return
    _render_tiles_from_cols("Sólidos (SS/SST)", cols, n_cols=4, interpret_numeric_as_status=False)

def render_dqo():
    cols = _filter_columns_by_keywords(cols_lower_noacc, KW_DQO)
    if not cols:
        return
    _render_tiles_from_cols("DQO", cols, n_cols=4, interpret_numeric_as_status=False)

def render_estados():
    cols = _filter_columns_by_keywords(cols_lower_noacc, KW_ESTADOS)
    if not
def render_estados():
    cols = _filter_columns_by_keywords(cols_lower_noacc, KW_ESTADOS)
    if not cols:
        return
    _render_tiles_from_cols("Estados", cols, n_cols=4, interpret_numeric_as_status=False)

# =========================
# GOOGLE SHEETS – ABA 2 (Químicos)
# =========================
GID_QMCS = "568520949"
CSV_URL_QMCS = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID_QMCS}"

df_qmcs = pd.read_csv(CSV_URL_QMCS)
df_qmcs.columns = [str(c).strip() for c in df_qmcs.columns]

# Corrigir nomes das colunas no df_qmcs para evitar erros
if "QUIMICOS" in df_qmcs.columns:
    df_qmcs.rename(columns={"QUIMICOS": "QUÍMICOS"}, inplace=True)

# =========================
# RENDERIZAÇÃO QUÍMICOS
# =========================
def render_quimicos():
    st.subheader("Químicos")

    # Exibir a tabela de químicos (última linha com dados)
    df_qmcs_last = df_qmcs.tail(1).copy()
    # Ajustar colunas para mostrar somente nomes sem acento em título
    df_qmcs_last.columns = [col.title() for col in df_qmcs_last.columns]

    st.dataframe(df_qmcs_last.T.rename(columns={df_qmcs_last.index[-1]: "Valor"}))

# =========================
# MAIN
# =========================
def main():
    st.title("Dashboard Operacional ETE")

    # Seção Caçambas (gauges)
    st.header("Caçambas")
    render_cacambas_gauges("Caçambas")

    # Seção Válvulas (cards)
    st.header("Válvulas")
    cols_valvulas = _filter_columns_by_keywords(cols_lower_noacc, KW_VALVULA)
    _render_tiles_from_cols("Válvulas", cols_valvulas, n_cols=4, interpret_numeric_as_status=True)

    # Seção Sopradores (cards)
    st.header("Sopradores")
    cols_sopradores = _filter_columns_by_keywords(cols_lower_noacc, KW_SOPRADOR)
    _render_tiles_from_cols("Sopradores", cols_sopradores, n_cols=4, interpret_numeric_as_status=True)

    # Seção Nitrificação e MBBR (cards com regra numérica OK)
    render_tiles_split_status("Status", [])

    # Seções adicionais
    render_outros_niveis()
    render_vazoes()
    render_ph()
    render_sst()
    render_dqo()
    render_estados()

    # Químicos (aba 2)
    render_quimicos()

if __name__ == "__main__":
    main()
