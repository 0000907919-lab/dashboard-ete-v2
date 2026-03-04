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
GID_FORM = "1283870792"

CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID_FORM}"

df = pd.read_csv(CSV_URL)
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
    return _strip_accents(str(s).lower()).replace(" ", "-").replace("/", "-")

cols_lower_noacc = [_strip_accents(c.lower()) for c in df.columns]
COLMAP = dict(zip(cols_lower_noacc, df.columns))

KW_CACAMBA = ["cacamba"]  # << apenas sem acento

# -------------------------
# Conversões e utilidades
# -------------------------
def to_float_ptbr(x):
    if isinstance(x, pd.Series):
        xx = x.dropna()
        x = xx.iloc[-1] if not xx.empty else np.nan
    elif isinstance(x, pd.DataFrame):
        xx = x.stack().dropna()
        x = xx.iloc[-1] if not xx.empty else np.nan
    elif isinstance(x, (list, tuple, np.ndarray)):
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
    if valid.empty:
        return None
    return valid.iloc[-1]# =========================
# GAUGES (somente Caçambas)
# =========================

def make_speedometer(val, label):
    nome = label
    if val is None or np.isnan(val):
        val = 0.0
    color = "#43A047" if val >= 70 else "#FB8C00" if val >= 30 else "#E53935"

    return go.Indicator(
        mode="gauge+number",
        value=float(val),
        number={"suffix": "%"},
        title={"text": f"<b>{nome}</b>", "font": {"size": 16}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}},
        domain={"x": [0, 1], "y": [0, 1]},
    )

# 🔧 NOVA FUNÇÃO — apenas colunas contendo “cacamba” viram gauge
def render_cacambas_gauges(title, n_cols=4):
    cols = [
        COLMAP[c]
        for c in cols_lower_noacc
        if ("cacamba" in c) and COLMAP[c] not in [None, ""]
    ]

    cols = sorted(cols)

    if not cols:
        st.info("Nenhuma caçamba encontrada.")
        return

    n_rows = int(np.ceil(len(cols) / n_cols))
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "indicator"}] * n_cols for _ in range(n_rows)],
        horizontal_spacing=0.05,
        vertical_spacing=0.15
    )

    for i, c in enumerate(cols):
        raw = last_valid_raw(df, c)
        val = to_float_ptbr(raw)
        r = i // n_cols + 1
        cc = i % n_cols + 1
        fig.add_trace(make_speedometer(val, c), row=r, col=cc)

    fig.update_layout(
        height=max(280 * n_rows, 280),
        margin=dict(l=10, r=10, t=10, b=10),
    )

    st.plotly_chart(fig, use_container_width=True, key=f"plot-gauges-{_slug(title)}")


# =========================
# TILES (cards genéricos)
# =========================

COLOR_OK = "#43A047"
COLOR_WARN = "#FB8C00"
COLOR_BAD = "#E53935"
COLOR_NEUTRAL = "#546E7A"
COLOR_NULL = "#9E9E9E"

def _units_from_label(label: str) -> str:
    s = _strip_accents(label.lower())
    if "m3/h" in s or "m³/h" in label.lower():
        return " m³/h"
    if "mg/l" in s:
        return " mg/L"
    if "(%)" in label or "%" in label:
        return "%"
    return ""

def _tile_color_and_text(raw_value, val_num, label, force_neutral_numeric=False):
    if raw_value is None:
        return COLOR_NULL, "—"

    t = _strip_accents(str(raw_value).strip().lower())
    if t in ["ok", "ligado", "aberto", "rodando", "on"]:
        return COLOR_OK, str(raw_value).upper()
    if t in ["nok", "falha", "erro", "fechado", "off"]:
        return COLOR_BAD, str(raw_value).upper()

    if not np.isnan(val_num):
        units = _units_from_label(label)
        base = _strip_accents(label.lower())

        # Vazão 0–200 m³/h
        if "vazao" in base or "vazão" in base:
            if 0 <= val_num <= 200:
                return COLOR_OK, f"{val_num:.0f} m³/h"
            else:
                return COLOR_BAD, f"{val_num:.0f} m³/h"

        # Semáforo numérico por regra
        from numpy import isnan
        color_by_rule = None
        if not force_neutral_numeric:
            color_by_rule = None  # (mantido simplificado)

        if color_by_rule is not None:
            return color_by_rule, f"{val_num:.2f}{units}"

        if force_neutral_numeric:
            return COLOR_NEUTRAL, f"{val_num:.2f}{units}"

        if units == "%":
            fill = COLOR_OK if val_num >= 70 else COLOR_WARN if val_num >= 30 else COLOR_BAD
            return fill, f"{val_num:.1f}%"

        return COLOR_NEUTRAL, f"{val_num:.2f}{units}"

    return COLOR_WARN, str(raw_value)


def _render_tiles_from_cols(title, cols_orig, n_cols=4, force_neutral_numeric=False):
    cols_orig = [c for c in cols_orig if c]
    cols_orig = sorted(cols_orig)

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
        fill, txt = _tile_color_and_text(
            raw, val, c, force_neutral_numeric=force_neutral_numeric
        )

        r = i // n_cols
        cc = i % n_cols

        x0, x1 = cc + 0.05, cc + 0.95
        y0, y1 = (n_rows - 1 - r) + 0.05, (n_rows - 1 - r) + 0.95

        fig.add_shape(
            type="rect",
            x0=x0, x1=x1, y0=y0, y1=y1,
            fillcolor=fill, line=dict(color="white", width=1)
        )

        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2 + 0.15,
            text=f"<b style='font-size:18px'>{txt}</b>",
            showarrow=False,
            font=dict(color="white")
        )

        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2 - 0.15,
            text=f"<span style='font-size:12px'>{c}</span>",
            showarrow=False,
            font=dict(color="white")
        )

    fig.update_layout(
        height=max(170 * n_rows, 170),
        margin=dict(l=10, r=10, t=10, b=10)
    )

    st.subheader(title)
    st.plotly_chart(fig, use_container_width=True, key=f"plot-tiles-{_slug(title)}")def render_tiles_split(title_base, base_keywords, n_cols=4, exclude_generic=True):
    excl = []  # simplificado
    # Nitrificação
    cols_nitr = []
    for norm in cols_lower_noacc:
        if any(k in norm for k in base_keywords) and "nitr" in norm:
            cols_nitr.append(COLMAP[norm])
    _render_tiles_from_cols(f"{title_base} – Nitrificação", cols_nitr, n_cols=n_cols)

    # MBBR
    cols_mbbr = []
    for norm in cols_lower_noacc:
        if any(k in norm for k in base_keywords) and "mbbr" in norm:
            cols_mbbr.append(COLMAP[norm])
    _render_tiles_from_cols(f"{title_base} – MBBR", cols_mbbr, n_cols=n_cols)


# =========================
# OUTROS GRUPOS
# =========================

def render_outros_niveis():
    cols = []
    for norm in cols_lower_noacc:
        if "nivel" in norm and "cacamba" not in norm:
            cols.append(COLMAP[norm])
    if cols:
        _render_tiles_from_cols("Níveis (MAB/TQ de Lodo)", cols, n_cols=3)

def render_vazoes():
    cols = []
    for norm in cols_lower_noacc:
        if "vazao" in norm:
            cols.append(COLMAP[norm])
    if cols:
        _render_tiles_from_cols("Vazões", cols, n_cols=3, force_neutral_numeric=True)

def render_ph():
    cols = []
    for norm in cols_lower_noacc:
        if "ph" in norm:
            cols.append(COLMAP[norm])
    if cols:
        _render_tiles_from_cols("pH", cols, n_cols=4)

def render_sst():
    cols = []
    for norm in cols_lower_noacc:
        if "sst" in norm or "ss " in norm:
            cols.append(COLMAP[norm])
    if cols:
        _render_tiles_from_cols("Sólidos (SS/SST)", cols, n_cols=4)

def render_dqo():
    cols = []
    for norm in cols_lower_noacc:
        if "dqo" in norm:
            cols.append(COLMAP[norm])
    if cols:
        _render_tiles_from_cols("DQO", cols, n_cols=4)

def render_estados():
    estados_kw = ["tridecanter", "desvio", "tempo de descarte", "volante"]
    cols = []
    for norm in cols_lower_noacc:
        if any(k in norm for k in estados_kw):
            cols.append(COLMAP[norm])
    if cols:
        _render_tiles_from_cols("Estados / Equipamentos", cols, n_cols=3)

# =========================
# CABEÇALHO – INFO
# =========================

def header_info():
    cand = ["carimbo de data/hora", "data", "operador"]
    found = {}
    for c in df.columns:
        k = _strip_accents(c.lower())
        for target in cand:
            if k == _strip_accents(target):
                found[target] = c

    col0, col1, col2 = st.columns(3)

    if "carimbo de data/hora" in found:
        col0.metric("Último carimbo", str(last_valid_raw(df, found["carimbo de data/hora"])))
    elif "data" in found:
        col0.metric("Data", str(last_valid_raw(df, found["data"])))

    if "operador" in found:
        col1.metric("Operador", str(last_valid_raw(df, found["operador"])))

    col2.metric("Registros", f"{len(df)} linhas")

# =========================
# DASHBOARD
# =========================

st.title("Dashboard Operacional ETE")
header_info()

# ---- GAUGES
render_cacambas_gauges("Caçambas")

# ---- Cards
render_tiles_split("Válvulas", ["valvula"])
render_tiles_split("Sopradores", ["soprador"])
render_tiles_split("Oxigenação", ["oxigenacao"], n_cols=4, exclude_generic=False)

render_outros_niveis()
render_vazoes()
render_ph()
render_sst()
render_dqo()
render_estados()

# =========================
# CARTAS DE CONTROLE — CUSTOS
# =========================

st.markdown("---")
st.header("🔴 Cartas de Controle — Custo (R$)")

with st.sidebar:
    gid_input = st.text_input("GID da aba de gastos", value="668859455")

CC_GID_GASTOS = gid_input.strip() or "668859455"
CC_URL_GASTOS = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={CC_GID_GASTOS}"

if st.button("🔄 Recarregar cartas"):
    st.rerun()

@st.cache_data(ttl=900, show_spinner=False)
def cc_baixar_csv_bruto(url: str, timeout: int = 20) -> pd.DataFrame:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    buf = io.StringIO(resp.text)
    df_txt = pd.read_csv(buf, dtype=str, keep_default_na=False, header=None)
    df_txt.columns = [str(c).strip() for c in df_txt.columns]
    return df_txtdef cc_strip_acc_lower(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()

def cc_find_header_row(df_txt: pd.DataFrame, max_scan: int = 120) -> int | None:
    kws_custo = ["custo", "custos", "gasto", "gastos", "valor", "$"]
    n = min(len(df_txt), max_scan)

    for i in range(n):
        row_vals = [cc_strip_acc_lower(x) for x in df_txt.iloc[i].tolist()]
        has_data = any("data" in v for v in row_vals)
        has_custo = any(any(kw in v for v in row_vals) for kw in kws_custo)
        if has_data and has_custo:
            return i
    return None

def cc_parse_currency_br(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.str.replace("\u00A0", " ", regex=False)
    s = s.str.replace("R$", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(".", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    s = s.apply(lambda x: re.sub(r"[^0-9.\-]", "", x))
    return pd.to_numeric(s, errors="coerce")

def cc_guess_item_label(df_txt, header_row, col_idx, fallback):
    label = ""
    if header_row - 1 >= 0:
        try:
            label = str(df_txt.iat[header_row - 1, col_idx]).strip()
        except:
            label = ""

    if not label:
        for j in range(col_idx - 1, max(-1, col_idx - 8), -1):
            try:
                v = str(df_txt.iat[header_row - 1, j]).strip()
            except:
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

with st.status("Carregando dados das cartas...", expanded=True) as status:
    try:
        st.write("• Baixando CSV do Google Sheets…")
        cc_df_raw = cc_baixar_csv_bruto(CC_URL_GASTOS)
        st.write(f"• CSV bruto: {cc_df_raw.shape[0]} linhas × {cc_df_raw.shape[1]} colunas")

        st.write("• Detectando linha de cabeçalho…")
        cc_hdr = cc_find_header_row(cc_df_raw)
        if cc_hdr is None:
            st.error("❌ Não achei a linha de cabeçalho com DATA e CUSTOS.")
            st.stop()

        cc_header_vals = [str(x).strip() for x in cc_df_raw.iloc[cc_hdr].tolist()]
        cc_df_all = cc_df_raw.iloc[cc_hdr + 1:].copy()
        cc_df_all.columns = cc_header_vals
        cc_df_all = cc_df_all.loc[:, [c.strip() != "" for c in cc_df_all.columns]]

        status.update(label="Dados carregados com sucesso ✅", state="complete")

    except Exception as e:
        st.error(f"❌ Erro ao preparar dados: {e}")
        st.stop()

cc_norm_cols = [cc_strip_acc_lower(c) for c in cc_df_all.columns]

CC_KW_COST_INCLUDE = ["custo", "custos", "gasto", "gastos", "valor", "$"]
CC_KW_COST_EXCLUDE = ["media", "média", "status", "automatic", "automatico", "automático", "meta"]

def cc_is_valid_cost_header(k):
    has_inc = any(x in k for x in CC_KW_COST_INCLUDE)
    has_exc = any(x in k for x in CC_KW_COST_EXCLUDE)
    return has_inc and not has_exc

cc_cost_idx_list = [i for i, nc in enumerate(cc_norm_cols) if cc_is_valid_cost_header(nc)]
cc_data_idx_list = [i for i, nc in enumerate(cc_norm_cols) if "data" in nc]

if not cc_cost_idx_list:
    st.error("❌ Não encontrei colunas de CUSTO válidas.")
    st.stop()

if not cc_data_idx_list:
    st.error("❌ Não encontrei coluna de DATA.")
    st.stop()

cc_items = []
cc_seen_labels = set()

for cost_idx in cc_cost_idx_list:
    cost_name = cc_df_all.columns[cost_idx]

    left_data = [i for i in cc_data_idx_list if i <= cost_idx]
    if left_data:
        data_idx = max(left_data)
    else:
        data_idx = min(cc_data_idx_list, key=lambda i: abs(i - cost_idx))

    data_name = cc_df_all.columns[data_idx]

    df_item = pd.DataFrame({
        "DATA": pd.to_datetime(cc_df_all.iloc[:, data_idx].astype(str),
                               errors="coerce", dayfirst=True),
        "CUSTO": cc_parse_currency_br(cc_df_all.iloc[:, cost_idx]),
    }).dropna(subset=["DATA", "CUSTO"]).sort_values("DATA")

    if df_item.empty:
        continue

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
    })

if not cc_items:
    st.warning("Nenhum item válido encontrado.")
    st.stop()

cc_labels_all = [it["label"] for it in cc_items]
cc_sel_labels = st.multiselect("Itens para exibir nas cartas", cc_labels_all, default=cc_labels_all)

cc_items = [it for it in cc_items if it["label"] in cc_sel_labels]

if not cc_items:
    st.info("Selecione pelo menos um item.")
    st.stop()

def cc_ultimo_valido_positivo(ser: pd.Series) -> float:
    s = pd.to_numeric(ser, errors="coerce")
    s = s[~s.isna()]
    if s.empty:
        return 0.0
    nz = s[s != 0]
    if not nz.empty:
        return float(nz.iloc[-1])
    return float(s.iloc[-1])

def cc_metricas_item(df_item):
    ultimo = cc_ultimo_valido_positivo(df_item["CUSTO"])

    mask_nz = df_item["CUSTO"].fillna(0) != 0
    idx_ref = mask_nz[mask_nz].index[-1] if mask_nz.any() else df_item.index[-1]

    iso_week = df_item["DATA"].dt.isocalendar()
    df_tmp = df_item.copy()
    df_tmp["__sem__"] = iso_week.week.astype(int)
    df_tmp["__anoiso__"] = iso_week.year.astype(int)

    ult_sem = int(df_tmp.loc[idx_ref, "__sem__"])
    ult_ano = int(df_tmp.loc[idx_ref, "__anoiso__"])

    custo_semana = df_tmp[
        (df_tmp["__sem__"] == ult_sem) &
        (df_tmp["__anoiso__"] == ult_ano)
    ]["CUSTO"].sum()

    df_tmp["__mes__"] = df_tmp["DATA"].dt.month
    df_tmp["__ano__"] = df_tmp["DATA"].dt.year

    ult_mes = int(df_tmp.loc[idx_ref, "__mes__"])
    ult_ano2 = int(df_tmp.loc[idx_ref, "__ano__"])

    custo_mes = df_tmp[
        (df_tmp["__mes__"] == ult_mes) &
        (df_tmp["__ano__"] == ult_ano2)
    ]["CUSTO"].sum()

    return ultimo, custo_semana, custo_mes

cc_tabs = st.tabs([it["label"] for it in cc_items])

for tab, it in zip(cc_tabs, cc_items):
    with tab:
        df_item = it["df"]
        ultimo, custo_semana, custo_mes = cc_metricas_item(df_item)

        c1, c2, c3 = st.columns(3)
        c1.metric("Custo do Dia", f"R$ {ultimo:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        c2.metric("Custo da Semana", f"R$ {custo_semana:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        c3.metric("Custo do Mês", f"R$ {custo_mes:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

        df_day = df_item.groupby("DATA", as_index=False)["CUSTO"].sum().sort_values("DATA")

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

        st.subheader("📅 Carta Diária")
        cc_desenhar_carta(df_day["DATA"], df_day["CUSTO"],
                          f"Custo Diário (R$) — {it['label']}", "R$")

        st.subheader("🗓️ Carta Semanal (ISO)")
        cc_desenhar_carta(df_week["Data"], df_week["CUSTO"],
                          f"Custo Semanal (R$) — {it['label']}", "R$")

        st.subheader("📆 Carta Mensal")
        cc_desenhar_carta(df_month["Data"], df_month["CUSTO"],
                          f"Custo Mensal (R$) — {it['label']}", "R$")

        with st.expander("🔍 Debug do item"):
            st.write("Coluna de DATA:", it["data_name"])
            st.write("Coluna de CUSTO:", it["cost_name"])
            st.dataframe(df_item.head(10))


# =========================
# RESUMO — SOPRADORES
# =========================

st.markdown("---")
st.subheader("🧾 Resumo — Sopradores (copiar e colar)")

def _col_matches_any(cnorm: str, kws):
    return any(k in cnorm for k in [_strip_accents(x.lower()) for x in kws])

def _select_soprador_cols(df_cols_norm, area_keywords):
    sel = []
    for c_norm in df_cols_norm:
        has_soprador = "soprador" in c_norm
        has_area = _col_matches_any(c_norm, area_keywords)
        if has_soprador and has_area:
            sel.append(COLMAP[c_norm])
    return sel

def _parse_status_ok_nok(raw):
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "—"
    t = _strip_accents(str(raw).strip().lower())
    if t in ["ok", "ligado", "aberto", "rodando", "on"]:
        return "OK"
    if t in ["nok", "falha", "erro", "fechado", "off"]:
        return "NOK"
    return "—"

def _extract_first_int(text: str):
    m = re.search(r"\d+", _strip_accents(text.lower()))
    return int(m.group()) if m else None

def _coletar_status_area(df, area_keywords):
    cols_area = _select_soprador_cols(cols_lower_noacc, area_keywords)
    itens = []
    for col in cols_area:
        num = _extract_first_int(col)
        raw = last_valid_raw(df, col)
        stt = _parse_status_ok_nok(raw)
        itens.append((num, stt, col))

    itens.sort(key=lambda x: (9999 if x[0] is None else x[0],
                               _strip_accents(x[2].lower())))

    pares = [f"{num} ({stt})" for num, stt, _ in itens if num is not None]
    return pares

def gerar_resumo_sopradores(df):
    mbbr_linha = _coletar_status_area(df, ["mbbr"])
    nitr_linha = _coletar_status_area(df, ["nitr"])

    linhas = []
    linhas.append("Sopradores MBBR:")
    linhas.append(" ".join(mbbr_linha) if mbbr_linha else "—")
    linhas.append("Sopradores Nitrificação:")
    linhas.append(" ".join(nitr_linha) if nitr_linha else "—")
    return "\n".join(linhas)

texto_resumo = gerar_resumo_sopradores(df)
st.text_area("Texto", value=texto_resumo, height=110, label_visibility="collapsed")

st.caption("Selecione e copie o texto acima para enviar no WhatsApp/relatório.")
