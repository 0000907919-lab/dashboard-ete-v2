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
 
cols_lower_noacc = [_strip_accents(c.lower()) for c in df.columns]
COLMAP = dict(zip(cols_lower_noacc, df.columns))  # normalizado -> original
 
# Palavras‑chave
KW_CACAMBA = ["cacamba", "caçamba"]
KW_NITR = ["nitr", "nitrificacao", "nitrificação"]
KW_MBBR = ["mbbr"]
KW_VALVULA = ["valvula", "válvula"]
KW_SOPRADOR = ["soprador", "oxigenacao", "oxigenação"]
 
# Grupos adicionais (puxar o que faltava)
KW_NIVEIS_OUTROS = ["nivel", "nível"]  # será filtrado excluindo caçamba
KW_VAZAO = ["vazao", "vazão"]
KW_PH = ["ph " , " ph"]      # espaços para evitar bater em 'oxipH' etc
KW_SST = ["sst ", " sst", "ss "]  # inclui SS/SST
KW_DQO = ["dqo " , " dqo"]
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
 
def _extract_number(base: str) -> str:
    return "".join(ch for ch in base if ch.isdigit())
 
def _remove_brackets(text: str) -> str:
    # Remove qualquer coisa após '['
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
def _nome_exibicao(label_original: str) -> str:
    """
    Padroniza nomes para:
      - "Nível da caçamba X"
      - "Soprador de nitrificação X" / "Soprador de MBBR X"
      - "Válvula de nitrificação X" / "Válvula de MBBR X"
      - Demais indicadores: remove colchetes e devolve texto limpo
    """
    base_clean = _remove_brackets(label_original)
    base = _strip_accents(base_clean.lower()).strip()
    num = _extract_number(base)
 
    # Caçambas
    if "cacamba" in base:
        return f"Nível da caçamba {num}" if num else "Nível da caçamba"
 
    # Sopradores (inclui Oxigenação)
    if ("soprador" in base) or ("oxigenacao" in base):
        if any(k in base for k in KW_NITR):
            return f"Soprador de nitrificação {num}" if num else "Soprador de nitrificação"
        if any(k in base for k in KW_MBBR):
            return f"Soprador de MBBR {num}" if num else "Soprador de MBBR"
        return f"Soprador {num}" if num else "Soprador"
 
    # Válvulas
    if "valvula" in base:
        if any(k in base for k in KW_NITR):
            return f"Válvula de nitrificação {num}" if num else "Válvula de nitrificação"
        if any(k in base for k in KW_MBBR):
            return f"Válvula de MBBR {num}" if num else "Válvula de MBBR"
        return f"Válvula {num}" if num else "Válvula"
 
    # Ajustes de capitalização comuns (pH, DQO, SST, Vazão, Nível, MIX)
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
 
def re_replace_case_insensitive(s, pattern, repl):
    import re
    return re.sub(pattern, repl, s, flags=re.IGNORECASE)
 
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
    # evita pegar colunas de sopradores/valvulas que por acaso tenham "caçamba"
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
# TILES (cards genéricos)
# =========================
def _tile_color_and_text(raw_value, val_num, label, force_neutral_numeric=False):
    """Define cor e texto do card conforme tipo de dado."""
    # Estados textuais
    if raw_value is None:
        return "#9E9E9E", "—"
 
    # numérico
    if not np.isnan(val_num):
        units = _units_from_label(label)
        if units == "%":
            # Percentuais com semáforo
            fill = "#43A047" if val_num >= 70 else "#FB8C00" if val_num >= 30 else "#E53935"
            return fill, f"{val_num:.1f}%"
        else:
            # Métricas de processo com cor neutra (pH, DQO, SST, Vazões, etc.)
            if force_neutral_numeric:
                return "#546E7A", f"{val_num:.2f}{units}"
            # Se não for neutro, usa mesma regra de semáforo
            fill = "#43A047" if val_num >= 70 else "#FB8C00" if val_num >= 30 else "#E53935"
            return fill, f"{val_num:.1f}{units}"
 
    # texto (OK/erro etc)
    txt = str(raw_value).strip()
    t = _strip_accents(txt.lower())
    if t in ["ok", "ligado", "aberto", "rodando", "on"]:
        return "#43A047", txt.upper()
    if t in ["nok", "falha", "erro", "fechado", "off"]:
        return "#E53935", txt.upper()
    return "#FB8C00", txt
 
def _render_tiles_from_cols(title, cols_orig, n_cols=4, force_neutral_numeric=False):
    cols_orig = [c for c in cols_orig if c]  # safe
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
        fill, txt = _tile_color_and_text(raw, val, c, force_neutral_numeric=force_neutral_numeric)
 
        r = i // n_cols
        cc = i % n_cols
        x0, x1 = cc + 0.05, cc + 0.95
        y0, y1 = (n_rows - 1 - r) + 0.05, (n_rows - 1 - r) + 0.95
 
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=fill, line=dict(color="white", width=1))
 
        nome = _nome_exibicao(c)
        # Valor
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2 + 0.15,
                           text=f"<b style='font-size:18px'>{txt}</b>",
                           showarrow=False, font=dict(color="white"))
        # Nome do item
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2 - 0.15,
                           text=f"<span style='font-size:12px'>{nome}</span>",
                           showarrow=False, font=dict(color="white"))
 
    fig.update_layout(height=max(170 * n_rows, 170),
                      margin=dict(l=10, r=10, t=10, b=10))
    st.subheader(title)
    st.plotly_chart(fig, use_container_width=True)
 
def render_tiles_split(title_base, base_keywords, n_cols=4):
    """Cards: Nitrificação e MBBR para Válvulas/Sopradores."""
    # Nitrificação
    cols_nitr = _filter_columns_by_keywords(cols_lower_noacc, base_keywords + KW_NITR)
    cols_nitr = [c for c in cols_nitr if not any(k in _strip_accents(c.lower()) for k in KW_CACAMBA)]
    _render_tiles_from_cols(f"{title_base} – Nitrificação", cols_nitr, n_cols=n_cols)
 
    # MBBR
    cols_mbbr = _filter_columns_by_keywords(cols_lower_noacc, base_keywords + KW_MBBR)
    cols_mbbr = [c for c in cols_mbbr if not any(k in _strip_accents(c.lower()) for k in KW_CACAMBA)]
    _render_tiles_from_cols(f"{title_base} – MBBR", cols_mbbr, n_cols=n_cols)
 
# -------------------------
# Grupos adicionais ("puxar o que faltava")
# -------------------------
def render_outros_niveis():
    # nível, mas não caçambas
    cols = _filter_columns_by_keywords(cols_lower_noacc, KW_NIVEIS_OUTROS)
    cols = [c for c in cols if not any(k in _strip_accents(c.lower()) for k in KW_CACAMBA)]
    if not cols:
        return
    # Para níveis com (%) seguimos semáforo; demais ficam neutros
    _render_tiles_from_cols("Níveis (MAB/TQ de Lodo)", cols, n_cols=3, force_neutral_numeric=False)
 
def render_vazoes():
    cols = _filter_columns_by_keywords(cols_lower_noacc, KW_VAZAO)
    if not cols:
        return
    _render_tiles_from_cols("Vazões", cols, n_cols=3, force_neutral_numeric=True)
 
def render_ph():
    cols = _filter_columns_by_keywords(cols_lower_noacc, KW_PH)
    if not cols:
        return
    _render_tiles_from_cols("pH", cols, n_cols=4, force_neutral_numeric=True)
 
def render_sst():
    cols = _filter_columns_by_keywords(cols_lower_noacc, KW_SST)
    if not cols:
        return
    _render_tiles_from_cols("Sólidos (SS/SST)", cols, n_cols=4, force_neutral_numeric=True)
 
def render_dqo():
    cols = _filter_columns_by_keywords(cols_lower_noacc, KW_DQO)
    if not cols:
        return
    _render_tiles_from_cols("DQO", cols, n_cols=4, force_neutral_numeric=True)
 
def render_estados():
    cols = _filter_columns_by_keywords(cols_lower_noacc, KW_ESTADOS)
    if not cols:
        return
    _render_tiles_from_cols("Estados / Equipamentos", cols, n_cols=3, force_neutral_numeric=False)
 
# =========================
# CABEÇALHO (última medição)
# =========================
def header_info():
    # tenta achar campos de auditoria
    cand = ["carimbo de data/hora", "data", "operador"]
    found = {}
    for c in df.columns:
        k = _strip_accents(c.lower())
        if k in [_strip_accents(x) for x in cand]:
            found[k] = c
 
    col0, col1, col2 = st.columns(3)
    if "carimbo de data/hora" in found:
        col0.metric("Último carimbo", str(last_valid_raw(df, found["carimbo de data/hora"])))
    elif "data" in found:
        col0.metric("Data", str(last_valid_raw(df, found["data"])))
    if "operador" in found:
        col1.metric("Operador", str(last_valid_raw(df, found["operador"])))
    # espaço reservado para algo adicional
    col2.metric("Registros", f"{len(df)} linhas")
 
# =========================
# DASHBOARD
# =========================
st.title("Dashboard Operacional ETE")
header_info()
 
# Caçambas (gauge)
render_cacambas_gauges("Caçambas")
 
# Válvulas (cards) — Nitrificação e MBBR
render_tiles_split("Válvulas", KW_VALVULA)
 
# Sopradores (cards) — Nitrificação e MBBR
render_tiles_split("Sopradores", KW_SOPRADOR)
 
# ---- Indicadores adicionais (o que estava faltando puxar)
render_outros_niveis()
render_vazoes()
render_ph()
# =========================================================
# 📊 CARTAS DE CONTROLE – MULTI QUÍMICOS (VERSÃO FINAL)
# =========================================================

st.markdown("---")
st.header("🔴 Cartas de Controle — Custos dos Químicos")

# URL da aba Seleção Químicos
GID_QUIM = "668859455"
URL_QUIM = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID_QUIM}"

# 1) Lê a aba completa (a primeira linha é a faixa azul com os nomes)
dfraw = pd.read_csv(URL_QUIM, header=None, dtype=str)

# Faixa azul com nomes dos produtos
linha_nomes = dfraw.iloc[0].tolist()

# Segunda linha contém os rótulos reais (DATA, CONSUMO, META, CUSTO $$ etc.)
_raw_header = dfraw.iloc[1].tolist()

# Normaliza rótulos (troca vazios por "COLUNA_VAZIA")
header_row = []
for h in _raw_header:
    if h is None or (isinstance(h, float) and pd.isna(h)):
        header_row.append("COLUNA_VAZIA")
    else:
        s = str(h).strip()
        header_row.append(s if s != "" else "COLUNA_VAZIA")

# Gera cabeçalhos únicos (evita erro de duplicados no Streamlit/Arrow)
def _make_unique(names):
    seen = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}__{seen[n]}")
    return out

header_unique = _make_unique(header_row)

# Dados começam na terceira linha em diante
dfq = dfraw.iloc[2:].copy()
dfq.columns = header_unique

# Remove colunas duplicadas definitivamente
dfq = dfq.loc[:, ~pd.Index(dfq.columns).duplicated()].reset_index(drop=True)

# Identifica colunas DATA e CUSTO $$ (ignora "Custo $$ azul")
colunas = [c.strip() for c in dfq.columns]
indices_data  = [i for i, c in enumerate(colunas) if c.upper() == "DATA"]
indices_custo = [i for i, c in enumerate(colunas) if c.upper() == "CUSTO $$"]

# Função para identificar corretamente o nome do químico
def _nome_quimico(idx_data_col):
    # tenta a mesma coluna
    try:
        nm = (linha_nomes[idx_data_col] or "").strip()
        if nm:
            return nm
    except:
        pass

    # varre para a esquerda (caso seja célula mesclada)
    j = idx_data_col - 1
    while j >= 0:
        val = (linha_nomes[j] or "").strip()
        if val:
            return val
        j -= 1

    return f"Químico col {idx_data_col}"

# Prepara dados por químico
dfs_quim = []

def preparar_dados_quimico(df, idx_data, idx_custo, nome):
    bloco = df[[df.columns[idx_data], df.columns[idx_custo]]].copy()
    bloco.columns = ["DATA", "CUSTO"]

    bloco["DATA"] = pd.to_datetime(bloco["DATA"], dayfirst=True, errors="coerce")

    bloco["CUSTO"] = (
        bloco["CUSTO"]
        .astype(str)
        .str.replace("R$", "")
        .str.replace(" ", "")
        .str.replace(".", "")
        .str.replace(",", ".", regex=False)
    )
    bloco["CUSTO"] = pd.to_numeric(bloco["CUSTO"], errors="coerce")

    bloco = bloco.dropna(subset=["DATA", "CUSTO"]).sort_values("DATA")
    bloco["Quimico"] = nome
    return bloco

# Monta cada bloco DATA → CUSTO
for idx_data in indices_data:
    candidatos = [i for i in indices_custo if i > idx_data]
    if not candidatos:
        continue

    idx_custo = candidatos[0]
    nome = _nome_quimico(idx_data)

    dfs_quim.append(preparar_dados_quimico(dfq, idx_data, idx_custo, nome))

# Se nada foi detectado
if not dfs_quim:
    st.error("Nenhum químico detectado na aba Seleção Químicos.")
    st.stop()

# Consolida
df_final = pd.concat(dfs_quim, ignore_index=True)

# Função universal das cartas
def desenhar_carta(x, y, titulo, ylabel):
    y = pd.Series(y).astype(float)
    media = y.mean()
    desvio = y.std(ddof=1) if len(y) > 1 else 0
    LSC = media + 3*desvio
    LIC = media - 3*desvio

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, y, marker="o", color="#1565C0")
    ax.axhline(media, color="blue", linestyle="--", label="Média")

    if desvio > 0:
        ax.axhline(LSC, color="red", linestyle="--", label="LSC +3σ")
        ax.axhline(LIC, color="red", linestyle="--", label="LIC -3σ")

        xs = pd.Series(x)
        ax.scatter(xs[y > LSC], y[y > LSC], color="red", marker="^", s=70)
        ax.scatter(xs[y < LIC], y[y < LIC], color="red", marker="v", s=70)

    ax.set_title(titulo)
    ax.set_xlabel("Data")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    st.pyplot(fig)

# =========================================================
# GERA AS CARTAS PARA CADA QUÍMICO
# =========================================================

for quim in df_final["Quimico"].unique():
    bloco = df_final[df_final["Quimico"] == quim]

    st.subheader(f"📌 {quim}")

    # Diário
    st.markdown("### 📅 Diário")
    desenhar_carta(bloco["DATA"], bloco["CUSTO"], f"Custo Diário — {quim}", "Custo (R$)")

    # Semanal
    df_week = (
        bloco.assign(semana=bloco["DATA"].dt.to_period("W-MON"))
             .groupby("semana", as_index=False)["CUSTO"].sum()
    )
    df_week["Data"] = df_week["semana"].dt.start_time

    st.markdown("### 🗓️ Semanal (ISO)")
    desenhar_carta(df_week["Data"], df_week["CUSTO"], f"Custo Semanal — {quim}", "Custo (R$)")

    # Mensal
    df_month = (
        bloco.assign(mes=bloco["DATA"].dt.to_period("M"))
             .groupby("mes", as_index=False)["CUSTO"].sum()
    )
    df_month["Data"] = df_month["mes"].dt.to_timestamp()

    st.markdown("### 📆 Mensal")
    desenhar_carta(df_month["Data"], df_month["CUSTO"], f"Custo Mensal — {quim}", "Custo (R$)")
``
render_sst()
render_dqo()
render_estados()
