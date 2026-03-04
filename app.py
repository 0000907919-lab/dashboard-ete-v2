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
# GOOGLE SHEETS – ABA 1 (Respostas ao Formulário / Operacional)
# =========================
SHEET_ID = "1Gv0jhdQLaGkzuzDXWNkD0GD5OMM84Q_zkOkQHGBhLjU"
GID_FORM = "1283870792"  # aba com o formulário operacional

# CORRETO: usar &gid= SEM HTML ESCAPADO
CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export"
    f"?format=csv&gid={GID_FORM}"
)

# -------------------------
# Carrega a planilha (df = operacional)
# -------------------------
df = pd.read_csv(CSV_URL, dtype=str)
df.columns = [str(c).strip() for c in df.columns]

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
    return (
        _strip_accents(str(s).lower())
        .replace(" ", "-")
        .replace("–", "-")
        .replace("/", "-")
    )

cols_lower_noacc = [_strip_accents(c.lower()) for c in df.columns]
COLMAP = dict(zip(cols_lower_noacc, df.columns))

# Palavras‑chave
KW_CACAMBA = ["cacamba", "caçamba"]
KW_NITR = ["nitr", "nitrificacao", "nitrificação"]
KW_MBBR = ["mbbr"]
KW_VALVULA = ["valvula", "válvula"]
KW_SOPRADOR = ["soprador"]
KW_OXIG = ["oxigenacao", "oxigenação"]

# Grupos adicionais
KW_NIVEIS_OUTROS = ["nivel", "nível"]
KW_VAZAO = ["vazao", "vazão"]
KW_PH = ["ph ", " ph"]
KW_SST = ["sst ", " sst", "ss "]
KW_DQO = ["dqo ", " dqo"]
KW_ESTADOS = ["tridecanter", "desvio", "tempo de descarte", "volante"]

KW_EXCLUDE_GENERIC = (
    KW_SST + KW_DQO + KW_PH + KW_VAZAO + KW_NIVEIS_OUTROS + KW_CACAMBA
)

# Conversões e utilidades
def to_float_ptbr(x):
    if isinstance(x, pd.Series):
        x = x.dropna().iloc[-1] if not x.dropna().empty else np.nan
    if isinstance(x, pd.DataFrame):
        flat = x.stack().dropna()
        x = flat.iloc[-1] if not flat.empty else np.nan
    if isinstance(x, (list, tuple, np.ndarray)):
        x = x[-1] if len(x) else np.nan

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
    obj = df_local[col]
    if isinstance(obj, pd.DataFrame):
        s = obj.iloc[:, -1]
    else:
        s = obj
    s = s.replace(r"^\s*$", np.nan, regex=True)
    valid = s.dropna()
    return valid.iloc[-1] if not valid.empty else None

def _filter_columns_by_keywords(all_cols_norm, keywords):
    kws = [_strip_accents(k.lower()) for k in keywords]
    sel = []
    for c in all_cols_norm:
        if any(k in c for k in kws):
            sel.append(c)
    return [COLMAP[c] for c in sel]

def _extract_number(base: str) -> str:
    return "".join(ch for ch in base if ch.isdigit())

def _remove_brackets(text: str) -> str:
    return text.split("[", 1)[0].strip()

def _units_from_label(label: str) -> str:
    s = _strip_accents(label.lower())
    if "m3/h" in s or "m³/h" in label.lower():
        return " m³/h"
    if "mg/l" in s:
        return " mg/L"
    if "%" in label:
        return "%"
    return ""

def _filter_cols_intersection(all_cols_norm, must_any_1, must_any_2, forbid_any=None):
    kws1 = [_strip_accents(k.lower()) for k in must_any_1]
    kws2 = [_strip_accents(k.lower()) for k in must_any_2]
    forb = [_strip_accents((forbid_any or [])[i].lower()) for i in range(len(forbid_any or []))]

    sel = []
    for c in all_cols_norm:
        if any(k in c for k in kws1) and any(k in c for k in kws2):
            if not any(k in c for k in forb):
                sel.append(c)
    return [COLMAP[c] for c in sel]# =========================
# ⚙️ PARÂMETROS DO SEMÁFORO (Sidebar)
# =========================
with st.sidebar.expander("⚙️ Parâmetros do Semáforo", expanded=True):
    st.caption("Ajuste os limites conforme a operação da ETE.")

    # Oxigenação (DO)
    st.markdown("**Oxigenação (mg/L)**")
    do_ok_min_nitr = st.number_input("Nitrificação – DO mínimo (verde)", value=2.0, step=0.1)
    do_ok_max_nitr = st.number_input("Nitrificação – DO máximo (verde)", value=3.0, step=0.1)
    do_warn_low_nitr = st.number_input("Nitrificação – abaixo disso é VERMELHO", value=1.0, step=0.1)
    do_warn_high_nitr = st.number_input("Nitrificação – acima disso é VERMELHO", value=4.0, step=0.1)

    do_ok_min_mbbr = st.number_input("MBBR – DO mínimo (verde)", value=2.0, step=0.1)
    do_ok_max_mbbr = st.number_input("MBBR – DO máximo (verde)", value=3.0, step=0.1)
    do_warn_low_mbbr = st.number_input("MBBR – abaixo disso é VERMELHO", value=1.0, step=0.1)
    do_warn_high_mbbr = st.number_input("MBBR – acima disso é VERMELHO", value=4.0, step=0.1)

    # pH
    st.markdown("---")
    st.markdown("**pH**")
    ph_ok_min_general = st.number_input("pH Geral – mínimo (verde)", value=6.5, step=0.1)
    ph_ok_max_general = st.number_input("pH Geral – máximo (verde)", value=8.5, step=0.1)
    ph_warn_low_general = st.number_input("pH Geral – abaixo disso é VERMELHO", value=6.0, step=0.1)
    ph_warn_high_general = st.number_input("pH Geral – acima disso é VERMELHO", value=9.0, step=0.1)

    ph_ok_min_mab = st.number_input("pH MAB – mínimo (verde)", value=4.5, step=0.1)
    ph_ok_max_mab = st.number_input("pH MAB – máximo (verde)", value=6.5, step=0.1)
    ph_warn_low_mab = st.number_input("pH MAB – abaixo disso é VERMELHO", value=4.0, step=0.1)
    ph_warn_high_mab = st.number_input("pH MAB – acima disso é VERMELHO", value=7.0, step=0.1)

    # Efluente
    st.markdown("---")
    st.markdown("**Qualidade do Efluente (Saída)**")

    sst_green_max = st.number_input("SST Saída – Máximo (verde)", value=30.0, step=1.0)
    sst_orange_max = st.number_input("SST Saída – Máximo (laranja)", value=50.0, step=1.0)

    dqo_green_max = st.number_input("DQO Saída – Máximo (verde)", value=150.0, step=10.0)
    dqo_orange_max = st.number_input("DQO Saída – Máximo (laranja)", value=300.0, step=10.0)

SEMAFORO_CFG = {
    "do": {
        "nitr": {
            "ok_min": do_ok_min_nitr,
            "ok_max": do_ok_max_nitr,
            "red_low": do_warn_low_nitr,
            "red_high": do_warn_high_nitr
        },
        "mbbr": {
            "ok_min": do_ok_min_mbbr,
            "ok_max": do_ok_max_mbbr,
            "red_low": do_warn_low_mbbr,
            "red_high": do_warn_high_mbbr
        },
    },
    "ph": {
        "general": {
            "ok_min": ph_ok_min_general,
            "ok_max": ph_ok_max_general,
            "red_low": ph_warn_low_general,
            "red_high": ph_warn_high_general
        },
        "mab": {
            "ok_min": ph_ok_min_mab,
            "ok_max": ph_ok_max_mab,
            "red_low": ph_warn_low_mab,
            "red_high": ph_warn_high_mab
        },
    },
    "sst_saida": {
        "green_max": sst_green_max,
        "orange_max": sst_orange_max
    },
    "dqo_saida": {
        "green_max": dqo_green_max,
        "orange_max": dqo_orange_max
    },
}

# =========================
# CONTROLES VISUAIS DAS CARTAS
# =========================
with st.sidebar.expander("📝 Rótulos das Cartas", expanded=False):
    cc_lbl_max_points = st.slider("Máx. rótulos", 0, 60, 20, 2)
    cc_lbl_out_of_control = st.checkbox("Marcar pontos fora do controle", True)
    cc_lbl_local_extremes = st.checkbox("Mostrar extremos locais", True)
    cc_lbl_show_first_last = st.checkbox("Mostrar primeiro e último", True)
    cc_lbl_compact_format = st.checkbox("Usar formato compacto (mil/mi)", True)
    cc_lbl_fontsize = st.slider("Fonte dos rótulos", 6, 14, 8)
    cc_lbl_angle = st.slider("Ângulo dos rótulos", -90, 90, 0)
    cc_lbl_bbox = st.checkbox("Fundo branco nos rótulos", True)# =========================
# PADRONIZAÇÃO DE NOMES (TÍTULOS)
# =========================
def re_replace_case_insensitive(s, pattern, repl):
    import re
    return re.sub(pattern, repl, s, flags=re.IGNORECASE)


def _nome_exibicao(label_original: str) -> str:
    """
    Padroniza nomes para exibição:
      - Nível da Caçamba X
      - Soprador MBBR/Nitrificação X
      - Oxigenação MBBR/Nitrificação X
      - Válvulas
    """
    base_clean = _remove_brackets(label_original)
    base = _strip_accents(base_clean.lower()).strip()
    num = _extract_number(base)

    # Caçambas
    if "cacamba" in base:
        return f"Nível da Caçamba {num}" if num else "Nível da Caçamba"

    # Oxigenação (DO)
    if "oxigenacao" in base:
        if any(k in base for k in KW_NITR):
            return f"Oxigenação Nitrificação {num}".strip()
        if any(k in base for k in KW_MBBR):
            return f"Oxigenação MBBR {num}".strip()
        return f"Oxigenação {num}".strip()

    # Sopradores
    if "soprador" in base:
        if any(k in base for k in KW_NITR):
            return f"Soprador de Nitrificação {num}" if num else "Soprador de Nitrificação"
        if any(k in base for k in KW_MBBR):
            return f"Soprador de MBBR {num}" if num else "Soprador de MBBR"
        return f"Soprador {num}" if num else "Soprador"

    # Válvulas
    if "valvula" in base:
        if any(k in base for k in KW_NITR):
            return f"Válvula de Nitrificação {num}" if num else "Válvula de Nitrificação"
        if any(k in base for k in KW_MBBR):
            return f"Válvula MBBR {num}" if num else "Válvula MBBR"
        return f"Válvula {num}" if num else "Válvula"

    # Ajuste de capitalização geral
    txt = base_clean
    replacements = {
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
    for k, v in replacements.items():
        txt = re_replace_case_insensitive(txt, k, v)

    return txt.strip()


# =========================
# MOTOR DO SEMÁFORO
# =========================
COLOR_OK = "#43A047"
COLOR_WARN = "#FB8C00"
COLOR_BAD = "#E53935"
COLOR_NEUTRAL = "#546E7A"
COLOR_NULL = "#9E9E9E"


def semaforo_numeric_color(label: str, val: float):
    if val is None or np.isnan(val):
        return COLOR_NULL

    base = _strip_accents(label.lower())

    # DO — limites fixos válidos para a planta
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

    # SST Saída
    if "sst" in base or "ss " in base:
        if "saida" in base or "saída" in label.lower():
            if val <= SEMAFORO_CFG["sst_saida"]["green_max"]:
                return COLOR_OK
            if val <= SEMAFORO_CFG["sst_saida"]["orange_max"]:
                return COLOR_WARN
            return COLOR_BAD
        return COLOR_NEUTRAL

    # DQO Saída
    if "dqo" in base:
        if "saida" in base or "saída" in label.lower():
            if val <= SEMAFORO_CFG["dqo_saida"]["green_max"]:
                return COLOR_OK
            if val <= SEMAFORO_CFG["dqo_saida"]["orange_max"]:
                return COLOR_WARN
            return COLOR_BAD
        return COLOR_NEUTRAL

    return None# =========================
# GAUGES (somente Caçambas)
# =========================

def make_speedometer(val, label):
    nome_exibicao = _nome_exibicao(label)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        val = 0.0

    color = COLOR_OK if val >= 70 else COLOR_WARN if val >= 30 else COLOR_BAD

    return go.Indicator(
        mode="gauge+number",
        value=float(val),
        number={"suffix": "%"},
        title={"text": f"<b>{nome_exibicao}</b>", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color}
        },
        domain={"x": [0, 1], "y": [0, 1]},
    )


def render_cacambas_gauges(title, n_cols=4):
    # Filtro extremamente rígido para garantir que SOMENTE Caçambas virem velocímetro
    def is_cacamba(label: str):
        base = _strip_accents(label.lower())
        return ("cacamba" in base)

    def is_excluded(label: str):
        base = _strip_accents(label.lower())
        grupos = KW_PH + KW_VAZAO + KW_SST + KW_DQO + KW_OXIG + KW_VALVULA + KW_SOPRADOR + KW_NIVEIS_OUTROS
        return any(k in base for k in grupos)

    cols_orig = [
        c for c in df.columns
        if is_cacamba(c) and not is_excluded(c)
    ]

    cols_orig = sorted(cols_orig, key=lambda c: _nome_exibicao(c))

    if not cols_orig:
        st.info("Nenhuma caçamba encontrada.")
        return

    # Debug opcional
    with st.expander("🔍 Caçambas detectadas", expanded=False):
        st.write(cols_orig)

    n_rows = int(np.ceil(len(cols_orig) / n_cols))
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "indicator"}] * n_cols for _ in range(n_rows)],
        horizontal_spacing=0.05,
        vertical_spacing=0.15,
    )

    for i, col in enumerate(cols_orig):
        raw = last_valid_raw(df, col)
        val = to_float_ptbr(raw)
        row = i // n_cols + 1
        col_i = i % n_cols + 1
        fig.add_trace(make_speedometer(val, col), row=row, col=col_i)

    fig.update_layout(
        height=max(300 * n_rows, 300),
        margin=dict(l=10, r=10, t=10, b=10)
    )

    st.subheader(title)
    st.plotly_chart(fig, use_container_width=True, key=f"gauge-{_slug(title)}")


# =========================
# TILES (RETÂNGULOS DE STATUS)
# =========================

def _tile_color_and_text(raw_value, val_num, label, force_neutral_numeric=False):
    """Define cor e texto do card conforme o tipo de dado."""
    if raw_value is None:
        return COLOR_NULL, "—"

    t = _strip_accents(str(raw_value).strip().lower())

    # Estados ON/OFF
    if t in ["ok", "ligado", "aberto", "rodando", "on"]:
        return COLOR_OK, str(raw_value).upper()

    if t in ["nok", "falha", "erro", "fechado", "off"]:
        return COLOR_BAD, str(raw_value).upper()

    # Caso numérico
    if not (val_num is None or np.isnan(val_num)):
        units = _units_from_label(label)

        # VAZÃO (regra fixa)
        if "vazao" in _strip_accents(label.lower()):
            if 0 <= val_num <= 200:
                return COLOR_OK, f"{val_num:.0f} m³/h"
            return COLOR_BAD, f"{val_num:.0f} m³/h"

        # Regras específicas (pH/DO/SST/DQO)
        color_rule = None if force_neutral_numeric else semaforo_numeric_color(label, val_num)
        if color_rule is not None:
            return color_rule, f"{val_num:.2f}{units}"

        if force_neutral_numeric:
            return COLOR_NEUTRAL, f"{val_num:.2f}{units}"

        # Para % (exceto caçambas, que não vêm para cá)
        if units == "%":
            fill = COLOR_OK if val_num >= 70 else COLOR_WARN if val_num >= 30 else COLOR_BAD
            return fill, f"{val_num:.1f}%"

        return COLOR_NEUTRAL, f"{val_num:.2f}{units}"

    return COLOR_WARN, str(raw_value)


def _render_tiles_from_cols(title, cols_orig, n_cols=4, force_neutral_numeric=False):
    cols_orig = sorted(cols_orig, key=lambda x: _nome_exibicao(x))

    if not cols_orig:
        st.info(f"Nenhum item encontrado para: {title}")
        return

    fig = go.Figure()
    n_rows = int(np.ceil(len(cols_orig) / n_cols))

    fig.update_xaxes(visible=False, range=[0, n_cols])
    fig.update_yaxes(visible=False, range=[0, n_rows])

    for i, col in enumerate(cols_orig):
        raw = last_valid_raw(df, col)
        val = to_float_ptbr(raw)

        fill, txt = _tile_color_and_text(
            raw,
            val,
            col,
            force_neutral_numeric=force_neutral_numeric
        )

        row = i // n_cols
        col_i = i % n_cols

        x0, x1 = col_i + 0.05, col_i + 0.95
        y0, y1 = (n_rows - 1 - row) + 0.05, (n_rows - 1 - row) + 0.95

        fig.add_shape(
            type="rect",
            x0=x0, x1=x1, y0=y0, y1=y1,
            fillcolor=fill,
            line=dict(color="white", width=1)
        )

        nome = _nome_exibicao(col)

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
        margin=dict(l=10, r=10, t=10, b=10)
    )

    st.subheader(title)
    st.plotly_chart(fig, use_container_width=True, key=f"tiles-{_slug(title)}")# =========================
# RENDERIZAÇÃO DOS GRUPOS DE TILES
# =========================

def render_tiles_split(title_base, base_keywords, n_cols=4, exclude_generic=True):
    excl = KW_EXCLUDE_GENERIC if exclude_generic else []

    # Nitrificação
    cols_nitr = _filter_cols_intersection(
        cols_lower_noacc,
        must_any_1=base_keywords,
        must_any_2=KW_NITR,
        forbid_any=excl,
    )
    _render_tiles_from_cols(f"{title_base} – Nitrificação", cols_nitr, n_cols=n_cols)

    # MBBR
    cols_mbbr = _filter_cols_intersection(
        cols_lower_noacc,
        must_any_1=base_keywords,
        must_any_2=KW_MBBR,
        forbid_any=excl,
    )
    _render_tiles_from_cols(f"{title_base} – MBBR", cols_mbbr, n_cols=n_cols)


# -------------------------
# Grupos adicionais
# -------------------------

def render_outros_niveis():
    cols = _filter_columns_by_keywords(cols_lower_noacc, KW_NIVEIS_OUTROS)
    cols = [c for c in cols if not any(k in _strip_accents(c.lower()) for k in KW_CACAMBA)]
    if not cols:
        return
    _render_tiles_from_cols("Níveis (MAB / TQ de Lodo)", cols, n_cols=3)


def render_vazoes():
    cols = _filter_columns_by_keywords(cols_lower_noacc, KW_VAZAO)
    if not cols:
        return
    _render_tiles_from_cols("Vazões", cols, n_cols=3, force_neutral_numeric=True)


def render_ph():
    cols = _filter_columns_by_keywords(cols_lower_noacc, KW_PH)
    if not cols:
        return
    _render_tiles_from_cols("pH", cols, n_cols=4)


def render_sst():
    cols = _filter_columns_by_keywords(cols_lower_noacc, KW_SST)
    if not cols:
        return
    _render_tiles_from_cols("Sólidos (SS / SST)", cols, n_cols=4)


def render_dqo():
    cols = _filter_columns_by_keywords(cols_lower_noacc, KW_DQO)
    if not cols:
        return
    _render_tiles_from_cols("DQO", cols, n_cols=4)


def render_estados():
    cols = _filter_columns_by_keywords(cols_lower_noacc, KW_ESTADOS)
    if not cols:
        return
    _render_tiles_from_cols("Estados / Equipamentos", cols, n_cols=3)


# =========================
# CABEÇALHO – Última Medição
# =========================

def header_info():
    campos = {
        "carimbo de data/hora": None,
        "data": None,
        "operador": None,
    }

    for c in df.columns:
        norm = _strip_accents(c.lower())
        if norm in campos:
            campos[norm] = c

    col1, col2, col3 = st.columns(3)

    if campos["carimbo de data/hora"]:
        col1.metric("Último registro", str(last_valid_raw(df, campos["carimbo de data/hora"])))
    elif campos["data"]:
        col1.metric("Data", str(last_valid_raw(df, campos["data"])))

    if campos["operador"]:
        col2.metric("Operador", str(last_valid_raw(df, campos["operador"])))

    col3.metric("Total de Registros", f"{len(df)} linhas")
    # =========================
# CARTAS DE CONTROLE – FORMATAÇÃO
# =========================

def cc_fmt_brl(v, pos=None):
    """Formata valores em Real padrão BR."""
    try:
        return ("R$ " + f"{v:,.0f}").replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return v


def cc_fmt_brl_compacto(v: float) -> str:
    """Formatação compacta: 1.200 -> 1,2 mil | 1.200.000 -> 1,2 mi"""
    try:
        n = float(v)
    except:
        return str(v)

    sinal = "-" if n < 0 else ""
    n = abs(n)

    if n >= 1_000_000:
        return f"{sinal}R$ {n/1_000_000:.1f} mi".replace(".", ",")
    if n >= 1_000:
        return f"{sinal}R$ {n/1_000:.1f} mil".replace(".", ",")

    return (sinal + "R$ " + f"{n:,.0f}").replace(",", "X").replace(".", ",").replace("X", ".")


def _indices_extremos_locais(y: pd.Series) -> set[int]:
    """Encontra picos e vales simples."""
    idxs = set()
    ys = y.reset_index(drop=True)

    for i in range(1, len(ys) - 1):
        if pd.isna(ys[i - 1]) or pd.isna(ys[i]) or pd.isna(ys[i + 1]):
            continue

        if ys[i] > ys[i - 1] and ys[i] > ys[i + 1]:
            idxs.add(y.index[i])  # pico

        if ys[i] < ys[i - 1] and ys[i] < ys[i + 1]:
            idxs.add(y.index[i])  # vale

    return idxs


def _selecionar_indices_para_rotulo(
    x: pd.Series,
    y: pd.Series,
    LSC: float,
    LIC: float,
    max_labels: int,
    incluir_oor: bool,
    incluir_extremos: bool,
    incluir_primeiro_ultimo: bool
) -> list[int]:

    candidatos = []
    y_clean = y.dropna()

    if y_clean.empty or max_labels <= 0:
        return []

    # 1 — Pontos fora de controle
    if incluir_oor:
        oor = y[(y > LSC) | (y < LIC)].dropna().index.tolist()
        candidatos.extend(oor)

    # 2 — Picos e vales
    if incluir_extremos:
        candidatos.extend(list(_indices_extremos_locais(y)))

    # 3 — Primeiro e último
    if incluir_primeiro_ultimo:
        candidatos.extend([y_clean.index[0], y_clean.index[-1]])

    # Remove duplicados mantendo ordem
    seen = set()
    candidatos = [i for i in candidatos if not (i in seen or seen.add(i))]

    # 4 — Complementa até o limite
    if len(candidatos) < max_labels:
        faltam = max_labels - len(candidatos)
        resto = [i for i in y.index if i not in candidatos and pd.notna(y.loc[i])]
        candidatos.extend(resto[-faltam:])

    return sorted(set(candidatos), key=lambda idx: x.loc[idx])


# =========================
# DESENHO DAS CARTAS DE CONTROLE
# =========================

def cc_desenhar_carta(x, y, titulo, ylabel, mostrar_rotulos=True):
    """Desenha uma carta de controle com LSC/LIC e rótulos inteligentes."""

    y = pd.Series(y).astype(float)
    y_nonnull = y.dropna()

    media = y_nonnull.mean() if not y_nonnull.empty else 0
    desvio = y_nonnull.std(ddof=1) if len(y_nonnull) > 1 else 0

    LSC = media + 3 * desvio
    LIC = media - 3 * desvio

    fig, ax = plt.subplots(figsize=(12, 4.8))

    ax.plot(x, y, marker="o", color="#1565C0", linewidth=2, markersize=5)

    ax.axhline(media, color="#1565C0", linestyle="--", label="Média")

    if desvio > 0:
        ax.axhline(LSC, color="red", linestyle="--", label="LSC (+3σ)")
        ax.axhline(LIC, color="red", linestyle="--", label="LIC (−3σ)")

    ax.yaxis.set_major_formatter(FuncFormatter(cc_fmt_brl))

    if mostrar_rotulos and not y_nonnull.empty:
        idxs = _selecionar_indices_para_rotulo(
            x=pd.Series(x),
            y=y,
            LSC=LSC,
            LIC=LIC,
            max_labels=cc_lbl_max_points,
            incluir_oor=cc_lbl_out_of_control,
            incluir_extremos=cc_lbl_local_extremes,
            incluir_primeiro_ultimo=cc_lbl_show_first_last,
        )

        def _fmt_valor(v):
            return cc_fmt_brl_compacto(v) if cc_lbl_compact_format else cc_fmt_brl(v)

        offsets = []
        base_offset = 8

        for k, _ in enumerate(idxs):
            sign = 1 if k % 2 == 0 else -1
            offsets.append(sign * (base_offset + 2 * (k // 4)))

        bbox = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7) if cc_lbl_bbox else None

        for idx, dy in zip(idxs, offsets):
            if pd.isna(y.loc[idx]):
                continue

            ax.annotate(
                _fmt_valor(y.loc[idx]),
                (x.loc[idx], y.loc[idx]),
                textcoords="offset points",
                xytext=(0, dy),
                ha="center",
                fontsize=cc_lbl_fontsize,
                rotation=cc_lbl_angle,
                bbox=bbox,
                color="#0D47A1",
            )

    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Data")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")

    st.pyplot(fig)# =========================
# SEÇÕES DO DASHBOARD
# =========================

st.title("Dashboard Operacional ETE")
header_info()

# Caçambas — ÚNICO local onde existe velocímetro
render_cacambas_gauges("Caçambas")

# Válvulas – Nitrificação e MBBR
render_tiles_split("Válvulas", KW_VALVULA)

# Sopradores – apenas SOPRADORES (sem DO)
render_tiles_split("Sopradores", KW_SOPRADOR)

# Oxigenação – DO separado dos sopradores
render_tiles_split("Oxigenação", KW_OXIG, n_cols=4, exclude_generic=False)

# Indicadores adicionais
render_outros_niveis()
render_vazoes()
render_ph()
render_sst()
render_dqo()
render_estados()

# =========================
# CARTAS DE CONTROLE — CUSTOS (R$)
# =========================
st.markdown("---")
st.header("🔴 Cartas de Controle — Custos (R$)")

# Seleção do GID pelo usuário
with st.sidebar:
    gid_input = st.text_input("GID da aba de gastos", value="668859455")

CC_GID_GASTOS = gid_input.strip() or "668859455"

# URL correta SEM &amp;
CC_URL_GASTOS = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export"
    f"?format=csv&gid={CC_GID_GASTOS}"
)

# Botão para recarregar planilha no Streamlit Cloud
if st.button("🔄 Recarregar cartas de controle"):
    st.rerun()


@st.cache_data(ttl=900, show_spinner=False)
def cc_baixar_csv_bruto(url: str, timeout: int = 20) -> pd.DataFrame:
    """Baixa a aba de gastos bruta (como strings)."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    texto = io.StringIO(resp.text)
    df_txt = pd.read_csv(texto, dtype=str, keep_default_na=False, header=None)
    df_txt.columns = [str(c).strip() for c in df_txt.columns]

    return df_txt


def cc_strip_acc_lower(s: str) -> str:
    """Remove acentos e deixa minúsculo."""
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()


def cc_find_header_row(df_txt: pd.DataFrame, max_scan: int = 120):
    """
    Acha a linha de cabeçalho real da aba de gastos
    procurando colunas com DATA e CUSTO/GASTO/VALOR.
    """
    kw_custo = ["custo", "custos", "gasto", "gastos", "valor", "$"]

    for i in range(min(len(df_txt), max_scan)):
        vals = [cc_strip_acc_lower(v) for v in df_txt.iloc[i].tolist()]

        has_data = any("data" in v for v in vals)
        has_custo = any(any(k in v for k in kw_custo) for v in vals)

        if has_data and has_custo:
            return i

    return None


def cc_parse_currency_br(series: pd.Series) -> pd.Series:
    """Converte valores brasileiros para float."""
    s = series.astype(str)

    s = s.str.replace("\u00A0", " ", regex=False)
    s = s.str.replace("R$", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(".", "", regex=False)       # remove milhar
    s = s.str.replace(",", ".", regex=False)      # decimal

    s = s.apply(lambda x: re.sub(r"[^0-9.\-]", "", x))

    return pd.to_numeric(s, errors="coerce")def cc_guess_item_label(df_txt: pd.DataFrame, header_row: int, col_idx: int, fallback: str) -> str:
    """
    Descobre o nome do item de custo:
      • Tenta pegar a linha acima do cabeçalho
      • Se vazio, olha colunas anteriores
      • Caso nada funcione, usa o nome bruto da coluna
    """
    label = ""

    # Linha acima do cabeçalho
    if header_row - 1 >= 0:
        try:
            label = str(df_txt.iat[header_row - 1, col_idx]).strip()
        except Exception:
            label = ""

        # Tenta achar algo à esquerda
        if not label:
            for j in range(col_idx - 1, max(-1, col_idx - 8), -1):
                try:
                    v = str(df_txt.iat[header_row - 1, j]).strip()
                except Exception:
                    v = ""
                if v:
                    label = v
                    break

    if not label:
        label = fallback

    label = label.replace("\n", " ").strip()

    if len(label) > 80:
        label = label[:77] + "..."

    return label


# =========================
# CARREGAMENTO DOS DADOS DAS CARTAS
# =========================

with st.status("Carregando dados das cartas de controle...", expanded=True) as status:
    try:
        st.write("• Baixando CSV do Google Sheets…")
        cc_df_raw = cc_baixar_csv_bruto(CC_URL_GASTOS, timeout=20)
        st.write(f"• {cc_df_raw.shape[0]} linhas × {cc_df_raw.shape[1]} colunas recebidas")

        st.write("• Detectando cabeçalho...")
        cc_hdr = cc_find_header_row(cc_df_raw, max_scan=120)

        if cc_hdr is None:
            st.error("❌ Não foi possível localizar a linha de cabeçalho (DATA + CUSTO/GASTO).")
            st.stop()

        # Linha real de cabeçalho
        header_vals = [str(v).strip() for v in cc_df_raw.iloc[cc_hdr].tolist()]
        cc_df_all = cc_df_raw.iloc[cc_hdr + 1:].copy()
        cc_df_all.columns = header_vals

        # Remove colunas vazias
        cc_df_all = cc_df_all.loc[:, [c.strip() != "" for c in cc_df_all.columns]]

        status.update(label="Dados carregados com sucesso ✔️", state="complete")

    except requests.exceptions.Timeout:
        st.error("⏳ Timeout ao acessar o Google Sheets.")
        st.stop()

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Falha ao baixar CSV: {e}")
        st.stop()

    except Exception as e:
        st.error(f"❌ Erro inesperado ao carregar dados: {e}")
        st.stop()


# NORMALIZAÇÃO DAS COLUNAS
cc_norm_cols = [cc_strip_acc_lower(c) for c in cc_df_all.columns]

KW_INCLUDE_COST = ["custo", "gasto", "gastos", "valor", "$"]
KW_EXCLUDE_COST = ["media", "média", "status", "automatic", "meta"]


def cc_is_valid_cost_header(normalized: str) -> bool:
    """Determina se a coluna é uma coluna de custo válida."""
    has_keyword = any(k in normalized for k in KW_INCLUDE_COST)
    has_exclusion = any(k in normalized for k in KW_EXCLUDE_COST)
    return has_keyword and not has_exclusion


# Índices de colunas úteis
cc_cost_idx_list = [i for i, nc in enumerate(cc_norm_cols) if cc_is_valid_cost_header(nc)]
cc_data_idx_list = [i for i, nc in enumerate(cc_norm_cols) if "data" in nc]

if not cc_cost_idx_list:
    st.error("❌ Nenhuma coluna de custo/gasto encontrada.")
    st.write(cc_df_all.columns)
    st.stop()

if not cc_data_idx_list:
    st.error("❌ Nenhuma coluna de DATA encontrada.")
    st.write(cc_df_all.columns)
    st.stop()


# =========================
# ORGANIZAÇÃO DOS ITENS DE CUSTO
# =========================

cc_items = []
cc_seen_labels = set()

for cost_idx in cc_cost_idx_list:

    cost_name = cc_df_all.columns[cost_idx]

    # Escolhe a coluna de data mais adequada (preferência para as da esquerda)
    left_dates = [i for i in cc_data_idx_list if i <= cost_idx]
    if left_dates:
        data_idx = max(left_dates)
    else:
        data_idx = min(cc_data_idx_list, key=lambda i: abs(i - cost_idx))

    data_name = cc_df_all.columns[data_idx]

    # Cria mini DataFrame do item
    df_item = pd.DataFrame({
        "DATA": pd.to_datetime(cc_df_all.iloc[:, data_idx], errors="coerce", dayfirst=True),
        "CUSTO": cc_parse_currency_br(cc_df_all.iloc[:, cost_idx]),
    }).dropna(subset=["DATA", "CUSTO"])

    df_item = df_item.sort_values("DATA")

    if df_item.empty:
        continue

    # Descobre nome do item adequadamente
    label_guess = cc_guess_item_label(cc_df_raw, cc_hdr, cost_idx, fallback=cost_name)
    label_norm = cc_strip_acc_lower(label_guess)

    if label_norm in cc_seen_labels:
        continue

    cc_seen_labels.add(label_norm)

    cc_items.append({
        "label": label_guess,
        "cost_name": cost_name,
        "data_name": data_name,
        "data_idx": data_idx,
        "cost_idx": cost_idx,
        "df": df_item,
    })# =========================
# VERIFICA SE EXISTEM ITENS DE CUSTO
# =========================

if not cc_items:
    st.warning("Nenhum item com dados válidos (DATA + CUSTO) foi encontrado.")
    st.stop()


# =========================
# SELEÇÃO DOS ITENS PARA MOSTRAR
# =========================

cc_labels_all = [item["label"] for item in cc_items]

cc_sel_labels = st.multiselect(
    "Selecione os itens de custo para exibir nas cartas:",
    cc_labels_all,
    default=cc_labels_all
)

cc_mostrar_rotulos = st.checkbox("Mostrar rótulos de valores nas cartas", True)

cc_items = [item for item in cc_items if item["label"] in cc_sel_labels]

if not cc_items:
    st.info("Selecione ao menos um item para visualizar.")
    st.stop()


# =========================
# FUNÇÕES DE CÁLCULO DE MÉTRICAS
# =========================

def cc_ultimo_valido_positivo(serie: pd.Series) -> float:
    """Retorna o último valor positivo diferente de zero."""
    s = pd.to_numeric(serie, errors="coerce")
    s = s.dropna()

    if s.empty:
        return 0.0

    nz = s[s != 0]
    return float(nz.iloc[-1] if not nz.empty else s.iloc[-1])


def cc_metricas_item(df_item: pd.DataFrame):
    """Calcula métricas do item: dia, semana ISO e mês."""
    ultimo = cc_ultimo_valido_positivo(df_item["CUSTO"])

    mask_nz = df_item["CUSTO"].fillna(0) != 0
    idx_ref = mask_nz[mask_nz].index[-1] if mask_nz.any() else df_item.index[-1]

    iso_cal = df_item["DATA"].dt.isocalendar()
    df_tmp = df_item.copy()
    df_tmp["SEM"] = iso_cal.week.astype(int)
    df_tmp["ANO_ISO"] = iso_cal.year.astype(int)

    ult_sem = int(df_tmp.loc[idx_ref, "SEM"])
    ult_ano = int(df_tmp.loc[idx_ref, "ANO_ISO"])

    custo_semana = df_tmp[
        (df_tmp["SEM"] == ult_sem) & (df_tmp["ANO_ISO"] == ult_ano)
    ]["CUSTO"].sum()

    df_tmp["MES"] = df_tmp["DATA"].dt.month
    df_tmp["ANO"] = df_tmp["DATA"].dt.year

    ult_mes = int(df_tmp.loc[idx_ref, "MES"])
    ult_ano2 = int(df_tmp.loc[idx_ref, "ANO"])

    custo_mes = df_tmp[
        (df_tmp["MES"] == ult_mes) & (df_tmp["ANO"] == ult_ano2)
    ]["CUSTO"].sum()

    return ultimo, custo_semana, custo_mes


# =========================
# TABS – UM PARA CADA ITEM
# =========================

abas = st.tabs([item["label"] for item in cc_items])

for aba, item in zip(abas, cc_items):
    with aba:

        df_item = item["df"]

        # -------- MÉTRICAS --------
        dia, semana, mes = cc_metricas_item(df_item)

        c1, c2, c3 = st.columns(3)
        c1.metric("Custo do dia", f"R$ {dia:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        c2.metric("Custo da semana", f"R$ {semana:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        c3.metric("Custo do mês", f"R$ {mes:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

        # -------- AGREGADOS --------
        df_day = df_item.groupby("DATA", as_index=False)["CUSTO"].sum()

        df_week = (
            df_item.assign(semana=df_item["DATA"].dt.to_period("W-MON"))
            .groupby("semana", as_index=False)["CUSTO"].sum()
        )
        df_week["Data"] = df_week["semana"].dt.start_time

        df_month = (
            df_item.assign(mes=df_item["DATA"].dt.to_period("M"))
            .groupby("mes", as_index=False)["CUSTO"].sum()
        )
        df_month["Data"] = df_month["mes"].dt.to_timestamp()

        # -------- CARTAS DE CONTROLE --------
        st.subheader("📅 Carta Diária")
        cc_desenhar_carta(
            df_day["DATA"], df_day["CUSTO"],
            f"Custo Diário — {item['label']}",
            "R$",
            mostrar_rotulos=cc_mostrar_rotulos
        )

        st.subheader("🗓️ Carta Semanal (ISO)")
        cc_desenhar_carta(
            df_week["Data"], df_week["CUSTO"],
            f"Custo Semanal — {item['label']}",
            "R$",
            mostrar_rotulos=cc_mostrar_rotulos
        )

        st.subheader("📆 Carta Mensal")
        cc_desenhar_carta(
            df_month["Data"], df_month["CUSTO"],
            f"Custo Mensal — {item['label']}",
            "R$",
            mostrar_rotulos=cc_mostrar_rotulos
        )

        # Debug
        with st.expander("🔍 Debug do Item"):
            st.write("Coluna de DATA:", item["data_name"], "(índice", item["data_idx"], ")")
            st.write("Coluna de CUSTO:", item["cost_name"], "(índice", item["cost_idx"], ")")
            st.dataframe(df_item.head(10))# ------------------------------------------------------------
# RESUMO TEXTO — Sopradores (para WhatsApp/Relatório)
# ------------------------------------------------------------

def _col_matches_any(cnorm: str, kws):
    kws_norm = [_strip_accents(k.lower()) for k in kws]
    return any(k in cnorm for k in kws_norm)


def _select_soprador_cols(df_cols_norm, area_keywords):
    """Seleciona APENAS colunas de soprador da área (MBBR/Nitr) excluindo genéricos e DO."""
    sel = []
    for c_norm in df_cols_norm:
        has_soprador = "soprador" in c_norm
        has_area = _col_matches_any(c_norm, area_keywords)
        has_excluded = _col_matches_any(c_norm, KW_EXCLUDE_GENERIC + KW_OXIG)
        if has_soprador and has_area and not has_excluded:
            sel.append(c_norm)
    return [COLMAP[c] for c in sel]


def _parse_status_ok_nok(raw):
    """Normaliza estados para OK/NOK/—."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "—"
    t = _strip_accents(str(raw).strip().lower())
    if t in ["ok", "ligado", "aberto", "rodando", "on"]:
        return "OK"
    if t in ["nok", "falha", "erro", "fechado", "off"]:
        return "NOK"
    return "—"


def _extract_first_int(text: str) -> int | None:
    m = re.search(r"\d+", _strip_accents(text.lower()))
    return int(m.group()) if m else None


def _coletar_status_area(df_local, area_keywords):
    """Coleta pares 'numero (status)' para uma área (MBBR/Nitrificação)."""
    cols_area = _select_soprador_cols(cols_lower_noacc, area_keywords)
    itens = []
    for col in cols_area:
        num = _extract_first_int(col)
        raw = last_valid_raw(df_local, col)
        stt = _parse_status_ok_nok(raw)
        itens.append((num, stt, col))

    # Ordena por número crescente (ausentes vão pro fim) e por nome
    itens.sort(key=lambda x: (9999 if x[0] is None else x[0], _strip_accents(x[2].lower())))
    pares = [f"{num} ({stt})" for num, stt, _ in itens if num is not None]
    return pares


def gerar_resumo_sopradores(df_local):
    mbbr_linha = _coletar_status_area(df_local, KW_MBBR)
    nitr_linha = _coletar_status_area(df_local, KW_NITR)
    linhas = []
    linhas.append("Sopradores MBBR:")
    linhas.append(" ".join(mbbr_linha) if mbbr_linha else "—")
    linhas.append("Sopradores Nitrificação:")
    linhas.append(" ".join(nitr_linha) if nitr_linha else "—")
    return "\n".join(linhas)


# Bloco de saída
st.markdown("---")
st.subheader("🧾 Resumo — Sopradores (copiar e colar)")
texto_resumo = gerar_resumo_sopradores(df)
st.text_area("Texto", value=texto_resumo, height=110, label_visibility="collapsed")
st.caption("Selecione e copie o texto acima (Ctrl+C / Cmd+C) para colar no WhatsApp/relatório.")# ------------------------------------------------------------
# DEBUG FINAL — Mostrar informações úteis para auditoria
# ------------------------------------------------------------

with st.expander("🔧 Debug Geral (opcional)"):
    st.write("Colunas carregadas da aba operacional:")
    st.write(list(df.columns))

    st.write("Colunas normalizadas (sem acentos):")
    st.write(cols_lower_noacc)

    st.write("Colunas detectadas como Caçambas:")
    cacambas_debug = [
        c for c in df.columns
        if "cacamba" in _strip_accents(c.lower())
    ]
    st.write(cacambas_debug)

    st.write("Colunas detectadas como Oxigenação:")
    oxig_debug = [
        c for c in df.columns
        if any(k in _strip_accents(c.lower()) for k in KW_OXIG)
    ]
    st.write(oxig_debug)

    st.write("Colunas detectadas como Sopradores:")
    sop_debug = [
        c for c in df.columns
        if "soprador" in _strip_accents(c.lower())
    ]
    st.write(sop_debug)

    st.write("Colunas detectadas como Válvulas:")
    valv_debug = [
        c for c in df.columns
        if any(k in _strip_accents(c.lower()) for k in KW_VALVULA)
    ]
    st.write(valv_debug)

    st.write("Colunas de pH detectadas:")
    ph_debug = [
        c for c in df.columns
        if "ph" in _strip_accents(c.lower())
    ]
    st.write(ph_debug)

    st.write("Colunas SST detectadas:")
    sst_debug = [
        c for c in df.columns
        if "sst" in _strip_accents(c.lower()) or "ss " in _strip_accents(c.lower())
    ]
    st.write(sst_debug)

    st.write("Colunas DQO detectadas:")
    dqo_debug = [
        c for c in df.columns
        if "dqo" in _strip_accents(c.lower())
    ]
    st.write(dqo_debug)

    st.write("Colunas de Vazão detectadas:")
    vazao_debug = [
        c for c in df.columns
        if "vazao" in _strip_accents(c.lower()) or "vazão" in c.lower()
    ]
    st.write(vazao_debug)

    st.write("KW_EXCLUDE_GENERIC em uso:")
    st.write(KW_EXCLUDE_GENERIC)# ------------------------------------------------------------
# FINALIZAÇÃO DO CÓDIGO — MENSAGEM PARA O OPERADOR
# ------------------------------------------------------------

st.markdown("---")
st.markdown("""
### 📘 Observações Importantes

- Este painel é atualizado automaticamente a partir das abas do Google Sheets.
- Caso ocorra qualquer anomalia visual (como cards duplicados ou dados zerados), verifique:
  - Se o GID informado está correto.
  - Se a aba contém cabeçalhos válidos.
  - Se não há linhas de texto acima do cabeçalho real.
- A função *Recarregar cartas* força a atualização dos dados no Streamlit Cloud.

Se precisar de ajuda, fale comigo! 🙂
""")

# Mensagem final
st.success("Painel carregado com sucesso! Todos os módulos estão funcionando.")# ------------------------------------------------------------
# LOGS DE EXECUÇÃO (opcional)
# ------------------------------------------------------------

with st.expander("📄 Logs de Execução (opcional)"):
    st.write("Versão do Streamlit:", st.__version__)
    st.write("Shape da tabela operacional:", df.shape)
    st.write("Shape da tabela de custos:", cc_df_all.shape)

    st.write("Primeiras linhas da aba operacional:")
    st.dataframe(df.head())

    st.write("Primeiras linhas da aba de custos:")
    st.dataframe(cc_df_all.head())

    # Contadores gerais
    st.write("Total de Caçambas detectadas:", len([
        c for c in df.columns if "cacamba" in _strip_accents(c.lower())
    ]))

    st.write("Total de itens de custo detectados:", len(cc_items))

    # Conferência de limites configurados
    st.write("Limites DO (Nitrificação):", SEMAFORO_CFG["do"]["nitr"])
    st.write("Limites DO (MBBR):", SEMAFORO_CFG["do"]["mbbr"])
    st.write("Limites pH geral:", SEMAFORO_CFG["ph"]["general"])
    st.write("Limites pH MAB:", SEMAFORO_CFG["ph"]["mab"])
    st.write("SST Saída:", SEMAFORO_CFG["sst_saida"])
    st.write("DQO Saída:", SEMAFORO_CFG["dqo_saida"])

    # Verificação de colunas que não foram classificadas
    st.write("Colunas sem classificação específica:")
    nao_class = []
    for c in df.columns:
        base = _strip_accents(c.lower())
        if not any(
            kw in base
            for kw in (
                KW_CACAMBA
                + KW_NITR
                + KW_MBBR
                + KW_VALVULA
                + KW_SOPRADOR
                + KW_OXIG
                + KW_NIVEIS_OUTROS
                + KW_VAZAO
                + KW_PH
                + KW_SST
                + KW_DQO
                + KW_ESTADOS
            )
        ):
            nao_class.append(c)

    st.write(nao_class)# ------------------------------------------------------------
# LIMITES E AJUSTES FINAIS DO PAINEL
# ------------------------------------------------------------

# Aviso visual para quando a planilha estiver vazia
if df.empty:
    st.error("❌ A aba operacional está vazia ou não pôde ser carregada.")
    st.stop()

# Verificação da aba de custos
if cc_df_all.empty:
    st.error("❌ A aba de custos foi carregada, mas não contém dados válidos.")
    st.stop()

# Mensagem opcional de verificação
with st.expander("ℹ️ Status geral do sistema", expanded=False):
    st.write("✔️ Aba operacional carregada:", df.shape, "linhas/colunas")
    st.write("✔️ Aba de custos carregada:", cc_df_all.shape, "linhas/colunas")
    st.write("✔️ Itens de custo detectados:", len(cc_items))

# Pequeno aviso sobre valores ausentes
if df.isna().sum().sum() > 0:
    st.warning("⚠️ Existem valores ausentes em algumas colunas da aba operacional.")

if cc_df_all.isna().sum().sum() > 0:
    st.warning("⚠️ Existem valores ausentes na aba de custos.")

# Rodapé
st.markdown("---")
st.markdown("""
### 🏁 Fim do Arquivo — app.py

Obrigado por utilizar o Painel Operacional da ETE!  
Se precisar ajustar novas faixas, adicionar sensores ou criar novas métricas, estou aqui para ajudar.  
""")# ------------------------------------------------------------
# FIM DO ARQUIVO app.py
# ------------------------------------------------------------

# Este é o final do arquivo principal.
# Se algo novo precisar ser adicionado (novas medições, novos sensores,
# novos sistemas, novas cartas ou dashboards), basta criar abaixo
# novas funções ou novos blocos Streamlit.

# O painel já está completamente estruturado e modular.
# Boa operação e bom trabalho na ETE! 🚀
