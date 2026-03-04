# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import unicodedata

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(page_title="Dashboard Operacional ETE", layout="wide")

# =========================
# GOOGLE SHEETS – ABA Operacional
# =========================
SHEET_ID = "1Gv0jhdQLaGkzuzDXWNkD0GD5OMM84Q_zkOkQHGBhLjU"
GID_FORM = "1283870792"  # aba com o formulário operacional

# >>> MUITO IMPORTANTE: &gid= (SEM &amp;)
CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export"
    f"?format=csv&gid={GID_FORM}"
)

@st.cache_data(ttl=300, show_spinner="Carregando dados operacionais...")
def carregar_dados_operacionais(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, dtype=str, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]
    return df

# Tenta carregar planilha
try:
    df = carregar_dados_operacionais(CSV_URL)
except Exception as e:
    st.error(f"Erro ao carregar planilha operacional: {e}")
    st.stop()

if df.empty:
    st.warning("A planilha operacional foi carregada, mas está vazia.")
    st.stop()

# =========================
# NORMALIZAÇÃO / AUXILIARES
# =========================
def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

def slug(s: str) -> str:
    return (
        strip_accents(str(s).lower())
        .replace(" ", "-")
        .replace("–", "-")
        .replace("/", "-")
    )

cols_lower_noacc = [strip_accents(c.lower()) for c in df.columns]
COLMAP = dict(zip(cols_lower_noacc, df.columns))

# Palavras‑chave
KW_CACAMBA       = ["cacamba", "caçamba"]
KW_NITR          = ["nitr", "nitrificacao", "nitrificação"]
KW_MBBR          = ["mbbr"]
KW_VALVULA       = ["valvula", "válvula"]
KW_SOPRADOR      = ["soprador"]
KW_OXIG          = ["oxigenacao", "oxigenação"]
KW_NIVEIS_OUTROS = ["nivel", "nível"]
KW_VAZAO         = ["vazao", "vazão"]
KW_PH            = ["ph ", " ph"]
KW_SST           = ["sst ", " sst", "ss "]
KW_DQO           = ["dqo ", " dqo"]
KW_ESTADOS       = ["tridecanter", "desvio", "tempo de descarte", "volante"]

KW_EXCLUDE_GENERIC = KW_SST + KW_DQO + KW_PH + KW_VAZAO + KW_NIVEIS_OUTROS + KW_CACAMBA

def to_float_ptbr(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip().replace("%", "")
    # 1.234,56 -> 1234.56 ; 10,5 -> 10.5
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    elif "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan

def last_valid_raw(df_local, col):
    s = df_local[col].replace(r"^\s*$", np.nan, regex=True)
    valid = s.dropna()
    return valid.iloc[-1] if not valid.empty else None

def filter_columns_by_keywords(all_cols_norm, keywords):
    kws = [strip_accents(k.lower()) for k in keywords]
    sel = [c for c in all_cols_norm if any(k in c for k in kws)]
    return [COLMAP[c] for c in sel]

def filter_cols_intersection(all_cols_norm, must_any_1, must_any_2, forbid_any=None):
    kws1 = [strip_accents(k.lower()) for k in must_any_1]
    kws2 = [strip_accents(k.lower()) for k in must_any_2]
    forb = [strip_accents(k.lower()) for k in (forbid_any or [])]
    sel = []
    for c in all_cols_norm:
        if any(k in c for k in kws1) and any(k in c for k in kws2):
            if not any(k in c for k in forb):
                sel.append(c)
    return [COLMAP[c] for c in sel]

def extract_number(base: str) -> str:
    return "".join(ch for ch in base if ch.isdigit())

def remove_brackets(text: str) -> str:
    return text.split("[", 1)[0].strip()

def units_from_label(label: str) -> str:
    s = strip_accents(label.lower())
    if "m3/h" in s or "m³/h" in label.lower():
        return " m³/h"
    if "mg/l" in s:
        return " mg/L"
    if "%" in label:
        return "%"
    return ""

def nome_exibicao(label_original: str) -> str:
    base_clean = remove_brackets(label_original)
    base = strip_accents(base_clean.lower()).strip()
    num = extract_number(base)

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
    txt = base_clean
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
# MOTOR DE COR (semafórico)
# =========================
COLOR_OK = "#43A047"      # verde
COLOR_WARN = "#FB8C00"    # laranja
COLOR_BAD = "#E53935"     # vermelho
COLOR_NEUTRAL = "#546E7A" # cinza azulado
COLOR_NULL = "#9E9E9E"    # cinza (sem dado)

def semaforo_numeric_color(label: str, val: float):
    if val is None or np.isnan(val):
        return COLOR_NULL
    base = strip_accents(label.lower())

    # Oxigenação DO (faixa fixa 1–5 mg/L)
    if "oxigenacao" in base:
        return COLOR_OK if 1 <= val <= 5 else COLOR_BAD

    # pH
    if re.search(r"\bph\b", base):
        cfg = SEMAFORO_CFG["ph"]["mab" if "mab" in base else "general"]
        if val < cfg["red_low"] or val > cfg["red_high"]:
            return COLOR_BAD
        if cfg["ok_min"] <= val <= cfg["ok_max"]:
            return COLOR_OK
        return COLOR_WARN

    # SST / SS — Saída
    if "sst" in base or re.search(r"\bss\b", base):
        if "saida" in base or "saída" in label.lower():
            if val <= SEMAFORO_CFG["sst_saida"]["green_max"]:
                return COLOR_OK
            if val <= SEMAFORO_CFG["sst_saida"]["orange_max"]:
                return COLOR_WARN
            return COLOR_BAD
        return COLOR_NEUTRAL

    # DQO — Saída
    if "dqo" in base:
        if "saida" in base or "saída" in label.lower():
            if val <= SEMAFORO_CFG["dqo_saida"]["green_max"]:
                return COLOR_OK
            if val <= SEMAFORO_CFG["dqo_saida"]["orange_max"]:
                return COLOR_WARN
            return COLOR_BAD
        return COLOR_NEUTRAL

    return None
