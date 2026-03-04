# -*- coding: utf-8 -*-
"""
Dashboard Operacional ETE — app.py

Características principais:
- Carrega dados da aba operacional do Google Sheets (CSV com &gid= correto)
- Caçambas exibidas em GAUGE (velocímetro) — apenas elas
- Demais indicadores em CARDS retangulares com semáforo (pH, SST/SS, DQO, Vazão, Oxigenação, Sopradores, Válvulas, Estados)
- Parâmetros do semáforo configuráveis na sidebar
- Cartas de controle (custos) diárias/semanais/mensais com rótulos inteligentes (opcional)
- Funções seguras (last_value) para evitar IndexError quando a coluna estiver vazia
- Debug/Status opcionais

Requisitos (pip): streamlit plotly matplotlib pandas requests
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import unicodedata
import io
import re
import requests

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(page_title="Dashboard Operacional ETE", layout="wide")

# =========================
# CONFIG GOOGLE SHEETS
# =========================
SHEET_ID = "1Gv0jhdQLaGkzuzDXWNkD0GD5OMM84Q_zkOkQHGBhLjU"
GID_FORM = "1283870792"  # aba com o formulário operacional

CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export"
    f"?format=csv&gid={GID_FORM}"
)

# =========================
# CARREGAMENTO DOS DADOS
# =========================
@st.cache_data(ttl=300, show_spinner="Carregando dados operacionais...")
def load_operacional(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    return df

try:
    df = load_operacional(CSV_URL)
except Exception as e:
    st.error(f"Erro ao carregar planilha operacional: {e}")
    st.stop()

if df.empty:
    st.warning("A planilha operacional foi carregada, mas está vazia.")
    st.stop()

# =========================
# NORMALIZAÇÃO / AUXILIARES
# =========================
def strip_acc(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

def slug(s: str) -> str:
    return (
        strip_acc(str(s).lower())
        .replace(" ", "-")
        .replace("–", "-")
        .replace("/", "-")
    )

cols_norm = [strip_acc(c.lower()) for c in df.columns]
COLMAP = dict(zip(cols_norm, df.columns))

# Palavras‑chave
KW_CACAMBA       = ["cacamba", "caçamba"]
KW_NITR          = ["nitr", "nitrificacao", "nitrificação"]
KW_MBBR          = ["mbbr"]
KW_VALVULA       = ["valvula", "válvula"]
KW_SOPRADOR      = ["soprador"]
KW_OXIG          = ["oxigenacao", "oxigenação"]
KW_NIVEIS_OUTROS = ["nivel", "nível"]
KW_VAZAO         = ["vazao", "vazão"]
KW_PH            = ["ph ", " ph", " ph ", "ph"]
KW_SST           = ["sst ", " sst", "ss ", "sst"]
KW_DQO           = ["dqo ", " dqo", "dqo"]
KW_ESTADOS       = ["tridecanter", "desvio", "tempo de descarte", "volante"]

KW_EXCLUDE_GENERIC = KW_SST + KW_DQO + KW_PH + KW_VAZAO + KW_NIVEIS_OUTROS + KW_CACAMBA

# Conversões e seleção

def to_float_ptbr(x):
    if x is None:
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


def last_value(col_name: str):
    s = df[col_name].replace(r"^\s*$", np.nan, regex=True).dropna()
    if s.empty:
        return None
    return s.iloc[-1]


def filter_columns_by_keywords(all_cols_norm, keywords):
    kws = [strip_acc(k.lower()) for k in keywords]
    sel = [c for c in all_cols_norm if any(k in c for k in kws)]
    return [COLMAP[c] for c in sel]


def filter_cols_intersection(all_cols_norm, must_any_1, must_any_2, forbid_any=None):
    kws1 = [strip_acc(k.lower()) for k in must_any_1]
    kws2 = [strip_acc(k.lower()) for k in must_any_2]
    forb = [strip_acc(k.lower()) for k in (forbid_any or [])]
    sel = []
    for c in all_cols_norm:
        if any(k in c for k in kws1) and any(k in c for k in kws2):
            if not any(k in c for k in forb):
                sel.append(c)
    return [COLMAP[c] for c in sel]


def units_from_label(label: str) -> str:
    s = strip_acc(label.lower())
    if "m3/h" in s or "m³/h" in label.lower():
        return " m³/h"
    if "mg/l" in s:
        return " mg/L"
    if "%" in label:
        return "%"
    return ""


def nome_exibicao(label_original: str) -> str:
    base = strip_acc(label_original.lower()).strip()
    num = "".join(ch for ch in label_original if ch.isdigit())
    if "cacamba" in base:
        return f"Nível da Caçamba {num}" if num else "Nível da Caçamba"
    if "oxigenacao" in base:
        if any(k in base for k in KW_NITR):
            return f"Oxigenação Nitrificação {num}".strip()
        if any(k in base for k in KW_MBBR):
            return f"Oxigenação MBBR {num}".strip()
        return f"Oxigenação {num}".strip()
    if "soprador" in base:
        if any(k in base for k in KW_NITR):
            return f"Soprador de Nitrificação {num}" if num else "Soprador de Nitrificação"
        if any(k in base for k in KW_MBBR):
            return f"Soprador de MBBR {num}" if num else "Soprador de MBBR"
        return f"Soprador {num}" if num else "Soprador"
    if "valvula" in base:
        if any(k in base for k in KW_NITR):
            return f"Válvula de Nitrificação {num}" if num else "Válvula de Nitrificação"
        if any(k in base for k in KW_MBBR):
            return f"Válvula MBBR {num}" if num else "Válvula MBBR"
        return f"Válvula {num}" if num else "Válvula"
    # Capitalização básica
    txt = label_original
    rep = {
        "ph": "pH",
        "sst": "SST",
        "ss ": "SS ",
        "dqo": "DQO",
        "vazao": "Vazão",
        "nivel": "Nível",
        "nível": "Nível",
        "mbbr": "MBBR",
        "nitrificacao": "Nitrificação",
        "mab": "MAB",
    }
    for k, v in rep.items():
        txt = re.sub(k, v, txt, flags=re.IGNORECASE)
    return txt.strip()

# =========================
# PARÂMETROS DO SEMÁFORO (Sidebar)
# =========================
with st.sidebar.expander("Parâmetros do Semáforo", expanded=True):
    st.caption("Ajuste os limites conforme a operação da ETE.")

    st.markdown("**Oxigenação (mg/L)**")
    do_ok_min_nitr    = st.number_input("Nitrificação – DO mínimo (verde)", value=2.0, step=0.1)
    do_ok_max_nitr    = st.number_input("Nitrificação – DO máximo (verde)", value=3.0, step=0.1)
    do_warn_low_nitr  = st.number_input("Nitrificação – abaixo disso é VERMELHO", value=1.0, step=0.1)
    do_warn_high_nitr = st.number_input("Nitrificação – acima disso é VERMELHO", value=4.0, step=0.1)

    do_ok_min_mbbr    = st.number_input("MBBR – DO mínimo (verde)", value=2.0, step=0.1)
    do_ok_max_mbbr    = st.number_input("MBBR – DO máximo (verde)", value=3.0, step=0.1)
    do_warn_low_mbbr  = st.number_input("MBBR – abaixo disso é VERMELHO", value=1.0, step=0.1)
    do_warn_high_mbbr = st.number_input("MBBR – acima disso é VERMELHO", value=4.0, step=0.1)

    st.markdown("---")
    st.markdown("**pH**")
    ph_ok_min_general   = st.number_input("pH Geral – mínimo (verde)", value=6.5, step=0.1)
    ph_ok_max_general   = st.number_input("pH Geral – máximo (verde)", value=8.5, step=0.1)
    ph_warn_low_general = st.number_input("pH Geral – abaixo disso é VERMELHO", value=6.0, step=0.1)
    ph_warn_high_general= st.number_input("pH Geral – acima disso é VERMELHO", value=9.0, step=0.1)

    ph_ok_min_mab   = st.number_input("pH MAB – mínimo (verde)", value=4.5, step=0.1)
    ph_ok_max_mab   = st.number_input("pH MAB – máximo (verde)", value=6.5, step=0.1)
    ph_warn_low_mab = st.number_input("pH MAB – abaixo disso é VERMELHO", value=4.0, step=0.1)
    ph_warn_high_mab= st.number_input("pH MAB – acima disso é VERMELHO", value=7.0, step=0.1)

    st.markdown("---")
    st.markdown("**Qualidade do Efluente (Saída)**")
    sst_green_max  = st.number_input("SST Saída – Máximo (verde)",  value=30.0, step=1.0)
    sst_orange_max = st.number_input("SST Saída – Máximo (laranja)", value=50.0, step=1.0)
    dqo_green_max  = st.number_input("DQO Saída – Máximo (verde)",   value=150.0, step=10.0)
    dqo_orange_max = st.number_input("DQO Saída – Máximo (laranja)",  value=300.0, step=10.0)

SEMAFORO_CFG = {
    "do": {
        "nitr": {"ok_min": do_ok_min_nitr,   "ok_max": do_ok_max_nitr,
                 "red_low": do_warn_low_nitr, "red_high": do_warn_high_nitr},
        "mbbr": {"ok_min": do_ok_min_mbbr,   "ok_max": do_ok_max_mbbr,
                 "red_low": do_warn_low_mbbr, "red_high": do_warn_high_mbbr},
    },
    "ph": {
        "general": {"ok_min": ph_ok_min_general,   "ok_max": ph_ok_max_general,
                    "red_low": ph_warn_low_general, "red_high": ph_warn_high_general},
        "mab":     {"ok_min": ph_ok_min_mab,   "ok_max": ph_ok_max_mab,
                    "red_low": ph_warn_low_mab, "red_high": ph_warn_high_mab},
    },
    "sst_saida":  {"green_max": sst_green_max,  "orange_max": sst_orange_max},
    "dqo_saida":  {"green_max": dqo_green_max,  "orange_max": dqo_orange_max},
}

# =========================
# COLOR RULES
# =========================
COLOR_OK = "#43A047"
COLOR_WARN = "#FB8C00"
COLOR_BAD = "#E53935"
COLOR_NEUTRAL = "#546E7A"
COLOR_NULL = "#9E9E9E"


def semaforo_numeric_color(label: str, val: float):
    if val is None or np.isnan(val):
        return COLOR_NULL
    base = strip_acc(label.lower())
    if "oxigenacao" in base:
        return COLOR_OK if 1 <= val <= 5 else COLOR_BAD
    if re.search(r"\bph\b", base):
        cfg = SEMAFORO_CFG["ph"]["mab" if "mab" in base else "general"]
        if val < cfg["red_low"] or val > cfg["red_high"]:
            return COLOR_BAD
        if cfg["ok_min"] <= val <= cfg["ok_max"]:
            return COLOR_OK
        return COLOR_WARN
    if "sst" in base or re.search(r"\bss\b", base):
        if "saida" in base or "saída" in label.lower():
            if val <= SEMAFORO_CFG["sst_saida"]["green_max"]:
                return COLOR_OK
            if val <= SEMAFORO_CFG["sst_saida"]["orange_max"]:
                return COLOR_WARN
            return COLOR_BAD
        return COLOR_NEUTRAL
    if "dqo" in base:
        if "saida" in base or "saída" in label.lower():
            if val <= SEMAFORO_CFG["dqo_saida"]["green_max"]:
                return COLOR_OK
            if val <= SEMAFORO_CFG["dqo_saida"]["orange_max"]:
                return COLOR_WARN
            return COLOR_BAD
        return COLOR_NEUTRAL
    return None

# =========================
# GAUGES — SOMENTE CAÇAMBAS
# =========================

def make_speedometer(val, label):
    nome = nome_exibicao(label)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        val = 0.0
    color = COLOR_OK if val >= 70 else COLOR_WARN if val >= 30 else COLOR_BAD
    return go.Indicator(
        mode="gauge+number",
        value=float(val),
        number={"suffix": "%"},
        title={"text": f"<b>{nome}</b>", "font": {"size": 16}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}},
        domain={"x": [0, 1], "y": [0, 1]},
    )


def render_cacambas_gauges(title, n_cols=4):
    all_cols = list(df.columns)
    def _is_cacamba(lbl: str) -> bool:
        return "cacamba" in strip_acc(lbl.lower())
    def _is_excluded(lbl: str) -> bool:
        base = strip_acc(lbl.lower())
        grupos = KW_PH + KW_VAZAO + KW_SST + KW_DQO + KW_OXIG + KW_VALVULA + KW_SOPRADOR + KW_NIVEIS_OUTROS
        return any(k in base for k in grupos)
    cols = [c for c in all_cols if _is_cacamba(c) and not _is_excluded(c)]
    cols = sorted(cols, key=lambda x: nome_exibicao(x))
    if not cols:
        st.info("Nenhuma caçamba encontrada.")
        return
    n_rows = int(np.ceil(len(cols)/n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols, specs=[[{"type": "indicator"}]*n_cols for _ in range(n_rows)],
                        horizontal_spacing=0.05, vertical_spacing=0.15)
    for i, col in enumerate(cols):
        raw = last_value(col)
        val = to_float_ptbr(raw)
        r = i // n_cols + 1
        c = i % n_cols + 1
        fig.add_trace(make_speedometer(val, col), r, c)
    fig.update_layout(height=max(280*n_rows, 280), margin=dict(l=10, r=10, t=10, b=10))
    st.subheader(title)
    st.plotly_chart(fig, use_container_width=True, key=f"gauges-{slug(title)}")

# =========================
# TILES — CARDS RETANGULARES
# =========================

def tile_color_text(raw_value, val_num, label, force_neutral=False):
    if raw_value is None:
        return COLOR_NULL, "—"
    t = strip_acc(str(raw_value).strip().lower())
    if t in ["ok", "ligado", "aberto", "rodando", "on"]:
        return COLOR_OK, str(raw_value).upper()
    if t in ["nok", "falha", "erro", "fechado", "off"]:
        return COLOR_BAD, str(raw_value).upper()
    if not (val_num is None or np.isnan(val_num)):
        units = units_from_label(label)
        if "vazao" in strip_acc(label.lower()):
            return (COLOR_OK if 0 <= val_num <= 200 else COLOR_BAD), f"{val_num:.0f} m³/h"
        color_rule = None if force_neutral else semaforo_numeric_color(label, val_num)
        if color_rule is not None:
            return color_rule, f"{val_num:.2f}{units}"
        if force_neutral:
            return COLOR_NEUTRAL, f"{val_num:.2f}{units}"
        if units == "%":
            fill = COLOR_OK if val_num >= 70 else COLOR_WARN if val_num >= 30 else COLOR_BAD
            return fill, f"{val_num:.1f}%"
        return COLOR_NEUTRAL, f"{val_num:.2f}{units}"
    return COLOR_WARN, str(raw_value)


def render_tiles_from_cols(title, cols_orig, n_cols=4, force_neutral=False):
    cols = sorted(cols_orig, key=lambda x: nome_exibicao(x))
    if not cols:
        st.info(f"Nenhum item encontrado para: {title}")
        return
    fig = go.Figure()
    n_rows = int(np.ceil(len(cols)/n_cols))
    fig.update_xaxes(visible=False, range=[0, n_cols])
    fig.update_yaxes(visible=False, range=[0, n_rows])
    for i, c in enumerate(cols):
        raw = last_value(c)
        val = to_float_ptbr(raw)
        color, txt = tile_color_text(raw, val, c, force_neutral)
        r = i // n_cols
        cc = i % n_cols
        x0, x1 = cc+0.05, cc+0.95
        y0, y1 = (n_rows-1-r)+0.05, (n_rows-1-r)+0.95
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=color, line=dict(color="white", width=1))
        nome = nome_exibicao(c)
        fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2+0.15, text=f"<b style='font-size:18px'>{txt}</b>", showarrow=False, font=dict(color="white"))
        fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2-0.15, text=f"<span style='font-size:12px'>{nome}</span>", showarrow=False, font=dict(color="white"))
    fig.update_layout(height=max(170*n_rows, 170), margin=dict(l=10, r=10, t=10, b=10))
    st.subheader(title)
    st.plotly_chart(fig, use_container_width=True, key=f"tiles-{slug(title)}")


def render_tiles_split(title_base, base_keywords, n_cols=4, exclude_generic=True):
    excl = KW_EXCLUDE_GENERIC if exclude_generic else []
    cols_nitr = filter_cols_intersection(cols_norm, base_keywords, KW_NITR, excl)
    render_tiles_from_cols(f"{title_base} – Nitrificação", cols_nitr, n_cols)
    cols_mbbr = filter_cols_intersection(cols_norm, base_keywords, KW_MBBR, excl)
    render_tiles_from_cols(f"{title_base} – MBBR", cols_mbbr, n_cols)

# =========================
# CABEÇALHO — Última Medição
# =========================

def header_info():
    campos = {"carimbo de data/hora": None, "data": None, "operador": None}
    for c in df.columns:
        norm = strip_acc(c.lower())
        if norm in campos:
            campos[norm] = c
    col1, col2, col3 = st.columns(3)
    if campos["carimbo de data/hora"]:
        col1.metric("Último registro", str(last_value(campos["carimbo de data/hora"])))
    elif campos["data"]:
        col1.metric("Data", str(last_value(campos["data"])))
    if campos["operador"]:
        col2.metric("Operador", str(last_value(campos["operador"])))
    col3.metric("Registros", f"{len(df)} linhas")

# =========================
# LAYOUT — SEÇÕES DO DASHBOARD
# =========================

st.title("Dashboard Operacional ETE")
header_info()

# Caçambas (gauge)
render_cacambas_gauges("Caçambas")

# Válvulas — Nitrificação e MBBR
render_tiles_split("Válvulas", KW_VALVULA)

# Sopradores — apenas sopradores (sem DO)
render_tiles_split("Sopradores", KW_SOPRADOR)

# Oxigenação — DO separado
render_tiles_split("Oxigenação", KW_OXIG, n_cols=4, exclude_generic=False)

# Indicadores adicionais
render_tiles_from_cols("Níveis (MAB / TQ de Lodo)", [c for c in filter_columns_by_keywords(cols_norm, KW_NIVEIS_OUTROS) if not any(k in strip_acc(c.lower()) for k in KW_CACAMBA)], n_cols=3)
render_tiles_from_cols("Vazões", filter_columns_by_keywords(cols_norm, KW_VAZAO), n_cols=3, force_neutral=True)
render_tiles_from_cols("pH",     filter_columns_by_keywords(cols_norm, KW_PH),   n_cols=4)
render_tiles_from_cols("Sólidos (SS / SST)", filter_columns_by_keywords(cols_norm, KW_SST), n_cols=4)
render_tiles_from_cols("DQO",    filter_columns_by_keywords(cols_norm, KW_DQO),  n_cols=4)
render_tiles_from_cols("Estados / Equipamentos", filter_columns_by_keywords(cols_norm, KW_ESTADOS), n_cols=3)

st.markdown("---")
st.caption("Painel carregado com sucesso.")

# =========================
# (Opcional) CARTAS DE CONTROLE — CUSTOS
# =========================
with st.expander("Cartas de Controle — Custos (opcional)", expanded=False):
    with st.sidebar:
        gid_input = st.text_input("GID da aba de gastos (opcional)", value="")
    if gid_input.strip():
        CC_GID = gid_input.strip()
        CC_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={CC_GID}"

        @st.cache_data(ttl=900, show_spinner=False)
        def cc_read(url: str) -> pd.DataFrame:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            buf = io.StringIO(resp.text)
            df_txt = pd.read_csv(buf, dtype=str, keep_default_na=False)
            df_txt.columns = [c.strip() for c in df_txt.columns]
            return df_txt

        try:
            cc_raw = cc_read(CC_URL)
            st.write("Aba de custos carregada:", cc_raw.shape)
        except Exception as e:
            st.error(f"Falha ao carregar custos: {e}")
            cc_raw = None

        if cc_raw is not None and not cc_raw.empty:
            # Heurística simples: achar colunas de DATA e CUSTO
            norm = [strip_acc(c.lower()) for c in cc_raw.columns]
            try:
                c_data = next(c for c, n in zip(cc_raw.columns, norm) if "data" in n)
            except StopIteration:
                c_data = None
            try:
                c_custo = next(c for c, n in zip(cc_raw.columns, norm) if any(k in n for k in ["custo","gasto","valor","$"]))
            except StopIteration:
                c_custo = None

            if c_data and c_custo:
                tmp = pd.DataFrame({
                    "DATA": pd.to_datetime(cc_raw[c_data], errors="coerce", dayfirst=True),
                    "CUSTO": pd.to_numeric(cc_raw[c_custo].astype(str)
                                            .str.replace("R$","", regex=False)
                                            .str.replace(" ","", regex=False)
                                            .str.replace(".","", regex=False)
                                            .str.replace(",",".", regex=False), errors="coerce"),
                }).dropna().sort_values("DATA")

                if not tmp.empty:
                    st.write("Prévia custos:")
                    st.line_chart(tmp.set_index("DATA")[["CUSTO"]])
                else:
                    st.info("Não há dados válidos (DATA + CUSTO) depois da limpeza.")
            else:
                st.info("Não encontrei colunas claras de DATA e CUSTO na aba informada.")

# =========================
# DEBUG OPCIONAL
# =========================
with st.expander("🔧 Debug Geral (opcional)", expanded=False):
    st.write("Colunas operacionais (originais):", list(df.columns))
    st.write("Colunas normalizadas:", cols_norm)
    cac = [c for c in df.columns if "cacamba" in strip_acc(c.lower())]
    st.write("Caçambas detectadas:", cac)
