# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(page_title="Dashboard Operacional ETE", layout="wide")

# =========================
# GOOGLE SHEETS – ABA 1 (Respostas ao Formulário / Operacional)
# =========================
SHEET_ID = "1Gv0jhdQLaGkzuzDXWNkD0GD5OMM84Q_zkOkQHGBhLjU"
GID_FORM = "1283870792"  # aba com o formulário operacional
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID_FORM}"

# -------------------------
# Carrega a planilha (df = operacional)
# -------------------------
df = pd.read_csv(CSV_URL)
df.columns = [str(c).strip() for c in df.columns]

# =========================
# NORMALIZAÇÃO / AUXILIARES
# =========================
def _strip_accents(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def _slug(s: str) -> str:
    # gera chave curta para evitar IDs duplicados em gráficos (Plotly)
    return _strip_accents(str(s).lower()).replace(" ", "-").replace("–", "-").replace("/", "-")

cols_lower_noacc = [_strip_accents(c.lower()) for c in df.columns]
COLMAP = dict(zip(cols_lower_noacc, df.columns))  # normalizado -> original

# Palavras‑chave
KW_CACAMBA   = ["cacamba", "caçamba"]
KW_NITR      = ["nitr", "nitrificacao", "nitrificação"]
KW_MBBR      = ["mbbr"]
KW_VALVULA   = ["valvula", "válvula"]
KW_SOPRADOR  = ["soprador", "oxigenacao", "oxigenação"]

# Grupos adicionais
KW_NIVEIS_OUTROS = ["nivel", "nível"]      # será filtrado excluindo caçamba
KW_VAZAO         = ["vazao", "vazão"]
KW_PH            = ["ph ", " ph"]          # espaços para evitar bater em 'oxipH' etc
KW_SST           = ["sst ", " sst", "ss "]  # inclui SS/SST
KW_DQO           = ["dqo ", " dqo"]
KW_ESTADOS       = ["tridecanter", "desvio", "tempo de descarte", "volante"]

# -------------------------
# Conversões e utilidades
# -------------------------
def to_float_ptbr(x):
    """Converte string PT-BR (%, vírgula) para float."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("%", "")
    # "10,5" -> "10.5" ; "1.234,5" -> "1234.5"
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    elif "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan

def last_valid_raw(df_local, col):
    """Último valor não vazio de uma coluna."""
    s = df_local[col].replace(r"^\s*$", np.nan, regex=True)
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
def re_replace_case_insensitive(s, pattern, repl):
    import re
    return re.sub(pattern, repl, s, flags=re.IGNORECASE)

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
    # key única evita StreamlitDuplicateElementId
    st.plotly_chart(fig, use_container_width=True, key=f"plot-gauges-{_slug(title)}")

# =========================
# TILES (cards genéricos)
# =========================
def _tile_color_and_text(raw_value, val_num, label, force_neutral_numeric=False):
    """Define cor e texto do card conforme tipo de dado."""
    if raw_value is None:
        return "#9E9E9E", "—"

    if not np.isnan(val_num):
        units = _units_from_label(label)
        if units == "%":
            fill = "#43A047" if val_num >= 70 else "#FB8C00" if val_num >= 30 else "#E53935"
            return fill, f"{val_num:.1f}%"
        else:
            if force_neutral_numeric:
                return "#546E7A", f"{val_num:.2f}{units}"
            fill = "#43A047" if val_num >= 70 else "#FB8C00" if val_num >= 30 else "#E53935"
            return fill, f"{val_num:.1f}{units}"

    txt = str(raw_value).strip()
    t = _strip_accents(txt.lower())
    if t in ["ok", "ligado", "aberto", "rodando", "on"]:
        return "#43A047", txt.upper()
    if t in ["nok", "falha", "erro", "fechado", "off"]:
        return "#E53935", txt.upper()
    return "#FB8C00", txt

def _render_tiles_from_cols(title, cols_orig, n_cols=4, force_neutral_numeric=False):
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
        val = to_float_ptbr(raw)
        fill, txt = _tile_color_and_text(raw, val, c, force_neutral_numeric=force_neutral_numeric)

        r = i // n_cols
        cc = i % n_cols
        x0, x1 = cc + 0.05, cc + 0.95
        y0, y1 = (n_rows - 1 - r) + 0.05, (n_rows - 1 - r) + 0.95

        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=fill, line=dict(color="white", width=1))

        nome = _nome_exibicao(c)
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2 + 0.15,
                           text=f"<b style='font-size:18px'>{txt}</b>",
                           showarrow=False, font=dict(color="white"))
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2 - 0.15,
                           text=f"<span style='font-size:12px'>{nome}</span>",
                           showarrow=False, font=dict(color="white"))

    fig.update_layout(height=max(170 * n_rows, 170),
                      margin=dict(l=10, r=10, t=10, b=10))
    st.subheader(title)
    st.plotly_chart(fig, use_container_width=True, key=f"plot-tiles-{_slug(title)}")

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
    cols = _filter_columns_by_keywords(cols_lower_noacc, KW_NIVEIS_OUTROS)
    cols = [c for c in cols if not any(k in _strip_accents(c.lower()) for k in KW_CACAMBA)]
    if not cols:
        return
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
    col2.metric("Registros", f"{len(df)} linhas")

# =========================
# DASHBOARD (como estava)
# =========================
st.title("Dashboard Operacional ETE")
header_info()

# Caçambas (gauge)
render_cacambas_gauges("Caçambas")

# Válvulas (cards) — Nitrificação e MBBR
render_tiles_split("Válvulas", KW_VALVULA)

# Sopradores (cards) — Nitrificação e MBBR
render_tiles_split("Sopradores", KW_SOPRADOR)

# ---- Indicadores adicionais
render_outros_niveis()
render_vazoes()
render_ph()
render_sst()
render_dqo()
render_estados()
# ============================================================
#        CARTAS DE CONTROLE — CUSTO (R$)  [UMA POR ITEM]
# ============================================================
st.markdown("---")
st.header("🔴 Cartas de Controle — Custo (R$)")

# ---- CONFIG: GID da aba (pode alterar pela sidebar) ----
with st.sidebar:
    gid_input = st.text_input("GID da aba de gastos", value="668859455")
GID_GASTOS = gid_input.strip() or "668859455"
URL_GASTOS = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID_GASTOS}"

# Botão de recarregar
if st.button("🔄 Recarregar cartas"):
    st.rerun()

# ------------------------------------------------------------
# 1) CARREGAR CSV COMO TEXTO (SEM HEADER) PARA NÃO PERDER LINHAS
# ------------------------------------------------------------
df_raw = pd.read_csv(URL_GASTOS, dtype=str, keep_default_na=False, header=None)

# ------------------------------------------------------------
# 2) AUXILIARES
# ------------------------------------------------------------
def _strip_acc_lower(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()

def _find_header_row(df_txt: pd.DataFrame, max_scan: int = 80) -> int | None:
    """
    Encontra a linha do cabeçalho: deve conter 'data' E algum de
    {'custo','custos','gasto','gastos','valor','$'}.
    (corrigido o NameError: usamos 'kw' e 'v' corretamente)
    """
    kws_custo = ["custo", "custos", "gasto", "gastos", "valor", "$"]
    n = min(len(df_txt), max_scan)
    for i in range(n):
        row_vals = [_strip_acc_lower(x) for x in df_txt.iloc[i].tolist()]
        has_data  = any("data" in v for v in row_vals)
        has_custo = any(any(kw in v for v in row_vals) for kw in kws_custo)
        if has_data and has_custo:
            return i
    return None

def _parse_currency_br(series: pd.Series) -> pd.Series:
    import re
    s = series.astype(str)
    s = s.str.replace("\u00A0", " ", regex=False)  # NBSP
    s = s.str.replace("R$", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(".", "", regex=False)       # milhar
    s = s.str.replace(",", ".", regex=False)      # decimal
    s = s.apply(lambda x: re.sub(r"[^0-9.\-]", "", x))
    return pd.to_numeric(s, errors="coerce")

def _guess_item_label(df_txt: pd.DataFrame, header_row: int, col_idx: int, fallback: str) -> str:
    """
    Tenta descobrir o nome do item olhando a linha ANTERIOR ao cabeçalho
    no MESMO bloco (ex.: 'PAC (LITROS) – código ...').
    Se estiver vazio, busca à esquerda até achar algo. Cai no fallback se nada achar.
    """
    label = ""
    if header_row - 1 >= 0:
        label = str(df_txt.iat[header_row - 1, col_idx]).strip()
        if not label:
            # busca alguns passos à esquerda
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
    # enxuga o nome para ficar amigável
    label = label.replace("\n", " ").strip()
    if len(label) > 80:
        label = label[:77] + "..."
    return label

# ------------------------------------------------------------
# 3) DETECTAR LINHA DE CABEÇALHO E REDEFINIR COLUNAS
# ------------------------------------------------------------
hdr = _find_header_row(df_raw, max_scan=80)
if hdr is None:
    st.error("❌ Não achei a linha de cabeçalho com DATA e CUSTOS na aba informada.")
    st.stop()

header_vals = [str(x).strip() for x in df_raw.iloc[hdr].tolist()]
df_all = df_raw.iloc[hdr + 1:].copy()
df_all.columns = header_vals

# remove colunas cujo nome está vazio (header em branco)
df_all = df_all.loc[:, [c.strip() != "" for c in df_all.columns]]

# ------------------------------------------------------------
# 4) IDENTIFICAR TODOS OS ITENS (pares DATA + CUSTO do mesmo bloco)
# ------------------------------------------------------------
norm_cols = [_strip_acc_lower(c) for c in df_all.columns]

# todos os custos/valores
cost_idx_list = [i for i, nc in enumerate(norm_cols)
                 if ("custo" in nc or "custos" in nc or "gasto" in nc or "gastos" in nc or "valor" in nc or "$" in nc)]

# todas as datas
data_idx_list = [i for i, nc in enumerate(norm_cols) if "data" in nc]

if not cost_idx_list:
    st.error("❌ Não encontrei nenhuma coluna de CUSTO/GASTO/VALOR.")
    st.write("Colunas disponíveis:", list(df_all.columns))
    st.stop()
if not data_idx_list:
    st.error("❌ Não encontrei nenhuma coluna de DATA.")
    st.write("Colunas disponíveis:", list(df_all.columns))
    st.stop()

items = []  # lista de dicts: {label, df}
for cost_idx in cost_idx_list:
    cost_name = df_all.columns[cost_idx]

    # DATA mais próxima à esquerda; se não houver, a mais próxima (esquerda/direita)
    left_data = [i for i in data_idx_list if i <= cost_idx]
    if left_data:
        data_idx = max(left_data)
    else:
        data_idx = min(data_idx_list, key=lambda i: abs(i - cost_idx))
    data_name = df_all.columns[data_idx]

    # cria dataframe do item (seleção por POSIÇÃO evita 'duplicate keys')
    df_item = pd.DataFrame({
        "DATA": pd.to_datetime(df_all.iloc[:, data_idx].astype(str), errors="coerce", dayfirst=True),
        "CUSTO": _parse_currency_br(df_all.iloc[:, cost_idx]),
    })
    df_item = df_item.dropna(subset=["DATA", "CUSTO"]).sort_values("DATA")

    if df_item.empty:
        continue

    # descobrir rótulo do item olhando a linha superior ao cabeçalho
    label_guess = _guess_item_label(df_raw, hdr, cost_idx, fallback=cost_name)

    items.append({
        "label": label_guess,
        "cost_name": cost_name,
        "data_name": data_name,
        "data_idx": data_idx,
        "cost_idx": cost_idx,
        "df": df_item
    })

if not items:
    st.warning("Nenhum item com dados válidos (DATA + CUSTO) foi encontrado.")
    st.stop()

# ------------------------------------------------------------
# 5) UI: seleção de itens + toggle de rótulos
# ------------------------------------------------------------
labels_all = [it["label"] for it in items]
sel_labels = st.multiselect("Itens para exibir nas cartas", labels_all, default=labels_all)
mostrar_rotulos = st.checkbox("Mostrar rótulos de dados nas cartas", value=True)

# manter apenas selecionados
items = [it for it in items if it["label"] in sel_labels]

# ------------------------------------------------------------
# 6) FUNÇÕES DE CARTA, FORMATAÇÃO E MÉTRICAS
# ------------------------------------------------------------
from matplotlib.ticker import FuncFormatter

def _fmt_brl(v, pos=None):
    try:
        return ("R$ " + f"{v:,.0f}").replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return v

def desenhar_carta(x, y, titulo, ylabel, mostrar_rotulos=True):
    y = pd.Series(y).astype(float)
    media = y.mean()
    desvio = y.std(ddof=1) if len(y) > 1 else 0.0
    LSC = media + 3*desvio
    LIC = media - 3*desvio

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(x, y, marker="o", color="#1565C0", label="Série")
    ax.axhline(media, color="blue", linestyle="--", label="Média")

    if desvio > 0:
        ax.axhline(LSC, color="red", linestyle="--", label="LSC (+3σ)")
        ax.axhline(LIC, color="red", linestyle="--", label="LIC (−3σ)")
        acima = y > LSC
        abaixo = y < LIC
        ax.scatter(pd.Series(x)[acima], y[acima], color="red", marker="^", s=70, zorder=3)
        ax.scatter(pd.Series(x)[abaixo], y[abaixo], color="red", marker="v", s=70, zorder=3)

    # Eixo Y em moeda BR
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_brl))

    # Rótulos de dados
    if mostrar_rotulos:
        for xi, yi in zip(x, y):
            if pd.notna(yi):
                ax.annotate(
                    ("R$ " + f"{yi:,.0f}").replace(",", "X").replace(".", ",").replace("X", "."),
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=8,
                    color="#1565C0",
                )

    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Data")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    st.pyplot(fig)

def _ultimo_valido_positivo(ser: pd.Series) -> float:
    s = pd.to_numeric(ser, errors="coerce")
    s = s[~s.isna()]
    if s.empty:
        return 0.0
    nz = s[s != 0]
    if not nz.empty:
        return float(nz.iloc[-1])
    return float(s.iloc[-1])

def _metricas_item(df_item: pd.DataFrame):
    # Dia
    ultimo = _ultimo_valido_positivo(df_item["CUSTO"])

    # Semana/mês: referência é a última linha com custo > 0; se não houver, a última
    mask_nz = df_item["CUSTO"].fillna(0) != 0
    idx_ref = mask_nz[mask_nz].index[-1] if mask_nz.any() else df_item.index[-1]

    iso_week = df_item["DATA"].dt.isocalendar()
    df_tmp = df_item.copy()
    df_tmp["__sem__"]    = iso_week.week.astype(int)
    df_tmp["__anoiso__"] = iso_week.year.astype(int)
    ult_sem = int(df_tmp.loc[idx_ref, "__sem__"])
    ult_ano = int(df_tmp.loc[idx_ref, "__anoiso__"])
    custo_semana = df_tmp[(df_tmp["__sem__"] == ult_sem) & (df_tmp["__anoiso__"] == ult_ano)]["CUSTO"].sum()

    df_tmp["__mes__"] = df_tmp["DATA"].dt.month
    df_tmp["__ano__"] = df_tmp["DATA"].dt.year
    ult_mes  = int(df_tmp.loc[idx_ref, "__mes__"])
    ult_ano2 = int(df_tmp.loc[idx_ref, "__ano__"])
    custo_mes = df_tmp[(df_tmp["__mes__"] == ult_mes) & (df_tmp["__ano__"] == ult_ano2)]["CUSTO"].sum()

    return ultimo, custo_semana, custo_mes

# ------------------------------------------------------------
# 7) ABAS — UMA POR ITEM
# ------------------------------------------------------------
tabs = st.tabs([it["label"] for it in items])

for tab, it in zip(tabs, items):
    with tab:
        df_item = it["df"]

        # Métricas do item
        ultimo, custo_semana, custo_mes = _metricas_item(df_item)
        c1, c2, c3 = st.columns(3)
        c1.metric("Custo do Dia",    f"R$ {ultimo:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        c2.metric("Custo da Semana", f"R$ {custo_semana:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        c3.metric("Custo do Mês",    f"R$ {custo_mes:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

        # Agregações do item
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

        # Cartas
        st.subheader("📅 Carta Diária")
        desenhar_carta(df_day["DATA"], df_day["CUSTO"], f"Custo Diário (R$) — {it['label']}", "R$", mostrar_rotulos=mostrar_rotulos)

        st.subheader("🗓️ Carta Semanal (ISO)")
        desenhar_carta(df_week["Data"], df_week["CUSTO"], f"Custo Semanal (R$) — {it['label']}", "R$", mostrar_rotulos=mostrar_rotulos)

        st.subheader("📆 Carta Mensal")
        desenhar_carta(df_month["Data"], df_month["CUSTO"], f"Custo Mensal (R$) — {it['label']}", "R$", mostrar_rotulos=mostrar_rotulos)

        # Debug opcional
        with st.expander("🔍 Debug do item"):
            st.write("Coluna de DATA original:", it["data_name"], " | índice:", it["data_idx"])
            st.write("Coluna de CUSTO original:", it["cost_name"], " | índice:", it["cost_idx"])
            st.dataframe(df_item.head())
