# pages/3_🔎_Screener.py
from __future__ import annotations
import pandas as pd
import streamlit as st
import numpy as np
from core.ui import app_header, ticker_selector
from core.data import load_watchlists, download_bulk
from core.indicators import sma, rsi
# --- Watchlists (fonte única + debug) ---------------------------------------
# --- Watchlists + classe ----------------------------------------------------
from core.data import load_watchlists
wl = load_watchlists()

# versão usada só para invalidar caches quando as watchlists forem atualizadas
ver: int = int(st.session_state.get("watchlists_version", 0))

st.caption(
    f"Watchlists: {'override (memória)' if 'watchlists_override' in st.session_state else 'arquivo'} "
    f"• v{ver} • BR:{len(wl.get('BR_STOCKS',[]))} | FIIs:{len(wl.get('BR_FIIS',[]))} | "
    f"BR Div:{len(wl.get('BR_DIVIDEND',[]))} | US:{len(wl.get('US_STOCKS',[]))} | "
    f"Cripto:{len(wl.get('CRYPTO',[]))}"
)

CLASS_MAP = {
    "Brasil (Ações B3)":   wl.get("BR_STOCKS", []),
    "Brasil (FIIs)":       wl.get("BR_FIIS", []),
    "Brasil — Dividendos": wl.get("BR_DIVIDEND", []),
    "Brasil — Blue Chips": wl.get("BR_BLUE_CHIPS", []),
    "Brasil — Small Caps": wl.get("BR_SMALL_CAPS", []),
    "EUA (Ações US)":      wl.get("US_STOCKS", []),
    "EUA — Dividendos":    wl.get("US_DIVIDEND", []),
    "EUA — Blue Chips":    wl.get("US_BLUE_CHIPS", []),
    "EUA — Small Caps":    wl.get("US_SMALL_CAPS", []),
    "Criptos":             wl.get("CRYPTO", []),
}
classe = st.selectbox("Classe", list(CLASS_MAP.keys()), index=0)
symbols = CLASS_MAP[classe]

if not symbols:
    st.warning("Nenhum ativo nesta classe. Atualize as watchlists na Settings ou reduza filtros.")
    st.code({k: len(v) for k, v in CLASS_MAP.items()}, language="json")
    st.stop()

# --- Controles de período / intervalo (defina ANTES de chamar _bulk) -------
cP, cI = st.columns(2)
with cP:
    period = st.selectbox("Período", ["3mo", "6mo", "1y", "2y", "5y"], index=1, key="scr_period")
with cI:
    interval = st.selectbox("Intervalo", ["1d", "1wk"], index=0, key="scr_interval")

# --- Cache de download (usa 'ver' para quebrar cache quando listas mudam) ---
@st.cache_data(ttl=600)
def _bulk(period: str, interval: str, symbols_tuple: tuple, _version: int):
    from core.data import download_bulk
    # _version é propositalmente não usado; serve só para invalidar o cache
    return download_bulk(list(symbols_tuple), period=period, interval=interval)

# >>> AGORA sim, com 'period' e 'interval' definidos, chame o cache:
data = _bulk(period, interval, tuple(symbols), ver)



st.set_page_config(page_title="Screener", page_icon="🔎", layout="wide")
app_header("🔎 Screener", "Triagem multi-ativos (BR/US/Cripto) com métricas e filtros")



# --- UI helpers para critérios/legenda ---
def criteria_summary(period, interval, min_price, min_avg_vol, only_trend_up, n_syms):
    cols = st.columns(5)
    cols[0].metric("Período", period)
    cols[1].metric("Intervalo", interval)
    cols[2].metric("Preço mín.", f"{min_price:,.2f}")
    cols[3].metric("Vol. médio mín.", f"{min_avg_vol:,.0f}")
    cols[4].metric("Tendência (SMA50>200)", "ON" if only_trend_up else "OFF")
    st.caption(f"Ativos em processamento: **{n_syms}**")

def metrics_legend():
    legend = [
        ("Price", "Último preço de fechamento."),
        ("D1%", "Variação percentual em 1 dia."),
        ("D5%", "Variação aprox. de 5 pregões (ou 1 semana se intervalo = 1wk)."),
        ("M1%", "Variação aprox. de 1 mês (≈21 barras diárias ou 4 semanais)."),
        ("M6%", "Variação aprox. de 6 meses (≈126 diárias ou 26 semanais)."),
        ("Y1%", "Variação aprox. de 1 ano (≈252 diárias ou 52 semanais)."),
        ("VolAnn%", "Volatilidade anualizada dos retornos simples."),
        ("AvgVol", "Média dos últimos ~60 volumes (unidades negociadas)."),
        ("RSI14", "Índice de Força Relativa (Wilder) em 14 períodos."),
        ("TrendUp", "Verdadeiro se SMA50 > SMA200 (tendência primária em alta)."),
        ("Score", "Momentum médio (D5%, M1%, M6%) + bônus de 5 se 45≤RSI≤60.")
    ]
    with st.expander("ℹ️ Critérios & métricas (clique para ver)", expanded=False):
        for k, v in legend:
            st.markdown(f"**{k}** — {v}")
        st.markdown("---")
        st.markdown("**Fórmula do Score**:  \n"
                    "Score = média( D5%, M1%, M6% )  +  (5 se 45 ≤ RSI14 ≤ 60, senão 0)")


# =======================
# Sidebar: seleção da classe e controles
# =======================
watchlists = load_watchlists()

classes = {
    "Brasil (Ações B3)": "BR_STOCKS",
    "EUA (Ações US)": "US_STOCKS",
    "Criptos": "CRYPTO"
}
class_label = st.sidebar.selectbox("Classe", list(classes.keys()), index=0)
asset_class = classes[class_label]
symbols_all = watchlists.get(asset_class, [])

st.sidebar.caption(f"Total na classe: **{len(symbols_all)}**")

# Filtros
period = st.sidebar.selectbox("Período", ["3mo", "6mo", "1y", "2y"], index=1)
interval = st.sidebar.selectbox("Intervalo", ["1d", "1wk"], index=0)
min_price = st.sidebar.number_input("Preço mínimo", min_value=0.0, value=1.0, step=0.5)
min_avg_vol = st.sidebar.number_input("Volume médio mínimo (unid.)", min_value=0.0, value=100000.0, step=50000.0)
only_trend_up = st.sidebar.checkbox("Somente tendência de alta (SMA50>SMA200)", value=False)

# Limite para processamento (para não travar em máquinas fracas)
max_symbols = st.sidebar.slider("Máx. de ativos processados", 10, min(200, len(symbols_all)) if symbols_all else 10, 60, 10)

# Campo de busca
query = st.sidebar.text_input("Filtrar por símbolo (contém)")

# =======================
# Seleção efetiva
# =======================
symbols = symbols_all.copy()
if query:
    symbols = [s for s in symbols if query.upper() in s.upper()]
symbols = symbols[:max_symbols]

st.write(f"Processando **{len(symbols)}** ativos desta classe…")

# Mostra o que está sendo medido/filtrado
criteria_summary(period, interval, min_price, min_avg_vol, only_trend_up, len(symbols))
metrics_legend()


if not symbols:
    st.warning("Nenhum ativo para processar. Ajuste filtros/consulta.")
    st.stop()

# =======================
# Download em lote
# =======================
data_dict = download_bulk(symbols, period=period, interval=interval)
ver = st.session_state.get("watchlists_version", 0)

@st.cache_data(ttl=600)
def _bulk(period, interval, symbols_tuple, version):
    from core.data import download_bulk
    return download_bulk(list(symbols_tuple), period=period, interval=interval)

data = _bulk(period, interval, tuple(symbols), ver)


# =======================
# Funções de métricas
# =======================
def pct_change(series: pd.Series, n: int) -> float:
    if len(series) <= n:
        return np.nan
    return (series.iloc[-1] / series.iloc[-1-n] - 1.0) * 100.0

def ann_vol(series: pd.Series, freq: str = "1d") -> float:
    # volatilidade anualizada a partir de retornos simples diários/semanais
    ret = series.pct_change().dropna()
    if ret.empty:
        return np.nan
    scale = 252 if freq == "1d" else 52
    return float(ret.std() * np.sqrt(scale) * 100.0)

def avg_volume(df: pd.DataFrame) -> float:
    return float(df["Volume"].tail(60).mean()) if "Volume" in df.columns and not df["Volume"].tail(60).empty else np.nan

def compute_row(sym: str, df: pd.DataFrame) -> dict:
    if df is None or df.empty or "Close" not in df.columns:
        return {
            "Symbol": sym, "Price": np.nan, "D1%": np.nan, "D5%": np.nan, "M1%": np.nan, "M6%": np.nan, "Y1%": np.nan,
            "VolAnn%": np.nan, "AvgVol": np.nan, "RSI14": np.nan, "TrendUp": False, "Score": np.nan
        }
    s_close = df["Close"].dropna()
    if s_close.empty:
        return {
            "Symbol": sym, "Price": np.nan, "D1%": np.nan, "D5%": np.nan, "M1%": np.nan, "M6%": np.nan, "Y1%": np.nan,
            "VolAnn%": np.nan, "AvgVol": np.nan, "RSI14": np.nan, "TrendUp": False, "Score": np.nan
        }

    price = float(s_close.iloc[-1])
    # janelas aproximadas para 1m/6m/1y conforme periodicidade
    n_map = {"1d": {"5d":5, "1m":21, "6m":126, "1y":252},
             "1wk":{"5d":1, "1m":4,  "6m":26,  "1y":52}}
    nm = n_map.get(interval, n_map["1d"])

    sma50 = sma(s_close, 50).iloc[-1] if len(s_close) >= 50 else np.nan
    sma200 = sma(s_close, 200).iloc[-1] if len(s_close) >= 200 else np.nan
    trend_up = bool(sma50 > sma200) if not np.isnan(sma50) and not np.isnan(sma200) else False

    row = {
        "Symbol": sym,
        "Price": price,
        "D1%": pct_change(s_close, 1),
        "D5%": pct_change(s_close, nm["5d"]),
        "M1%": pct_change(s_close, nm["1m"]),
        "M6%": pct_change(s_close, nm["6m"]),
        "Y1%": pct_change(s_close, nm["1y"]),
        "VolAnn%": ann_vol(s_close, freq=interval),
        "AvgVol": avg_volume(df),
        "RSI14": float(rsi(s_close, 14).iloc[-1]) if len(s_close) >= 14 else np.nan,
        "TrendUp": trend_up,
    }
    # Score simples = momentum médio + bônus quando RSI entre 45-60
    mom_list = [x for x in [row["D5%"], row["M1%"], row["M6%"]] if pd.notna(x)]
    mom_mean = float(np.nanmean(mom_list)) if mom_list else np.nan
    rsi_bonus = 5.0 if 45 <= (row["RSI14"] or 0) <= 60 else 0.0
    row["Score"] = mom_mean + rsi_bonus if pd.notna(mom_mean) else np.nan
    return row

# =======================
# Monta DataFrame do screener
# =======================
rows = []
for sym, df in data_dict.items():
    r = compute_row(sym, df)
    rows.append(r)

screen = pd.DataFrame(rows)

# =======================
# Filtros pós-cálculo
# =======================
if min_price > 0:
    screen = screen[screen["Price"] >= min_price]
if min_avg_vol > 0:
    screen = screen[screen["AvgVol"] >= min_avg_vol]
if only_trend_up:
    screen = screen[screen["TrendUp"] == True]

# Ordenação
sort_by = st.selectbox(
    "Ordenar por",
    ["Score","M1%","M6%","Y1%","D1%","VolAnn%","AvgVol","Price","RSI14","Symbol"],
    index=0
)
ascending = st.toggle("Ordem crescente", value=False)
screen = screen.sort_values(by=sort_by, ascending=ascending, na_position="last")

# =======================
# Exibição
# =======================
st.caption("Dica: clique no cabeçalho para ordenar; use a barra lateral para filtrar.")
def _color_pct(val):
    if pd.isna(val): 
        return ""
    return "color: #0b8f43;" if val >= 0 else "color: #c23b22;"

def _color_score(val):
    if pd.isna(val): 
        return ""
    # verde leve até forte conforme pontuação
    if val >= 10: return "background-color: #e6f4ea; color: #0b8f43;"
    if val >= 5:  return "background-color: #f0fbf3; color: #0b8f43;"
    return ""

styled = (
    screen.style
    .applymap(_color_pct, subset=["D1%","D5%","M1%","M6%","Y1%"])
    .applymap(_color_score, subset=["Score"])
    .format({
        "Price": "{:,.2f}",
        "D1%": "{:+.2f}%", "D5%": "{:+.2f}%", "M1%": "{:+.2f}%", "M6%": "{:+.2f}%", "Y1%": "{:+.2f}%",
        "VolAnn%": "{:.1f}%", "AvgVol": "{:,.0f}", "RSI14": "{:.1f}", "Score": "{:+.2f}"
    })
)

# =======================
# Seleção de ativos (envio p/ Backtest)
# =======================
screen = screen.reset_index(drop=True)
if "Select" not in screen.columns:
    screen.insert(0, "Select", False)

st.caption("Marque os ativos que deseja enviar para o Backtest.")
edited = st.data_editor(
    screen,
    use_container_width=True,
    height=560,
    disabled=[],
    hide_index=True
)

selected_symbols = edited.loc[edited["Select"] == True, "Symbol"].tolist()
st.session_state["screener_selection"] = selected_symbols

c1, c2 = st.columns([1,1])
csv_bytes = edited.drop(columns=["Select"]).to_csv(index=False).encode("utf-8")
with c1:
    st.download_button(
        "⬇️ Exportar CSV",
        data=csv_bytes,
        file_name=f"screener_{asset_class.lower()}_{period}_{interval}.csv",
        mime="text/csv"
    )
with c2:
    st.success(f"Selecionados para Backtest: {len(selected_symbols)}")
    st.caption("Abra a aba **📊 Backtest** para usar a seleção.")

