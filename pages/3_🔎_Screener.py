# pages/3_🧪_Screener.py

from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import streamlit as st

# yfinance direto, para ficar autossuficiente
import yfinance as yf


# ======================================================================================
# Helpers robustos
# ======================================================================================

def _flatten_col(c) -> str:
    """
    Converte nomes de coluna em string plana.
    - Se c for tuple (MultiIndex), retorna a última parte não vazia
    - Caso contrário, str(c)
    """
    if isinstance(c, tuple):
        for part in reversed(c):
            if isinstance(part, str) and part.strip():
                return part
        # fallback
        return "_".join(str(p) for p in c)
    return str(c)


def _to_series(x) -> pd.Series:
    """
    Garante que x seja uma Series numérica (mesmo se vier escalar/array/list).
    """
    if x is None:
        return pd.Series(dtype="float64")
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce")
    # numpy array, list, ou escalar
    return pd.to_numeric(pd.Series(x), errors="coerce")


def _ret(series_like, periods: int) -> float:
    """
    Retorno percentual em 'periods' períodos, robusto para qualquer entrada.
    """
    s = _to_series(series_like).dropna()
    if s.size <= periods:
        return np.nan
    v = s.pct_change(periods=periods).iloc[-1]
    return float(v) if pd.notna(v) else np.nan


def _rsi(series_like, n: int = 14) -> float:
    """
    RSI de n períodos (Wilder), robusto.
    """
    s = _to_series(series_like).dropna()
    if s.size <= n:
        return np.nan
    delta = s.diff()
    up = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / down
    rsi = 100 - (100 / (1 + rs))
    v = rsi.iloc[-1]
    return float(v) if pd.notna(v) else np.nan


def _sma(series_like, win: int) -> float:
    s = _to_series(series_like).dropna()
    if s.size < win:
        return np.nan
    return float(s.rolling(win).mean().iloc[-1])


def _vol_ann(series_like) -> float:
    """
    Volatilidade anualizada (aprox) em %, robusto.
    """
    s = _to_series(series_like).dropna()
    if s.size < 2:
        return np.nan
    std = s.pct_change().std()
    if pd.isna(std):
        return np.nan
    return float(std * math.sqrt(252) * 100.0)


# Formatações para style
def _fmt_price(x):
    return "" if pd.isna(x) else f"{x:,.2f}"

def _fmt_pct(x):
    return "" if pd.isna(x) else f"{x*100:,.2f}%"

def _fmt_int(x):
    return "" if pd.isna(x) else f"{int(x):,}"


def _color_pct(v):
    if pd.isna(v):
        return ""
    return "color:#00b894" if v >= 0 else "color:#d63031"

def _color_score(v):
    if pd.isna(v):
        return ""
    # escala simples 0..4
    if v >= 3.0:
        return "background-color:#14532d;color:#eafff0"
    if v >= 2.0:
        return "background-color:#166534;color:#eafff0"
    if v >= 1.0:
        return "background-color:#15803d;color:#f0fff7"
    if v > 0.0:
        return "background-color:#22c55e;color:#052e13"
    return "background-color:#ef4444;color:#fff"


# ======================================================================================
# Cache de download em bloco (usa yfinance)
# ======================================================================================

@st.cache_data(show_spinner=False, ttl=60*30)  # 30 min
def download_bulk(symbols: Iterable[str], period: str, interval: str, ver: int = 0) -> Dict[str, pd.DataFrame]:
    """
    Baixa OHLCV para uma lista de símbolos de forma robusta.
    Retorna dict[symbol] -> DataFrame (index datetime, colunas ['Open','High','Low','Close','Adj Close','Volume'])
    """
    syms = list(dict.fromkeys([s.strip() for s in symbols if s and str(s).strip()]))  # únicos preservando ordem
    if not syms:
        return {}

    try:
        data = yf.download(
            tickers=" ".join(syms),
            period=period,
            interval=interval,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception:
        return {}

    out: Dict[str, pd.DataFrame] = {}

    # Quando vem 1 só ticker, o yfinance devolve DF simples (não MultiIndex)
    if isinstance(data.columns, pd.MultiIndex):
        # MultiIndex: (ticker, field)
        for s in syms:
            try:
                df = data[s].copy()
            except Exception:
                continue
            # normaliza colunas
            df = df.copy()
            df.columns = [_flatten_col(c).title() for c in df.columns]
            if "Date" in df.columns:
                df = df.set_index(pd.to_datetime(df["Date"], errors="coerce")).drop(columns=["Date"])
            df.index.name = None
            out[s] = df
    else:
        # Único ticker
        df = data.copy()
        df.columns = [_flatten_col(c).title() for c in df.columns]
        if "Date" in df.columns:
            df = df.set_index(pd.to_datetime(df["Date"], errors="coerce")).drop(columns=["Date"])
        df.index.name = None
        out[syms[0]] = df

    return out


# ======================================================================================
# Watchlists (carrega de core.data se existir; senão um fallback enxuto)
# ======================================================================================

def _friendly(name_key: str) -> str:
    """
    Converte chave técnica para um nome amigável.
    """
    mapping = {
        "BR_STOCKS": "Brasil (Ações B3)",
        "BR_FIIS": "Brasil (FIIs)",
        "BR_DIVIDEND": "Brasil — Dividendos",
        "BR_BLUE_CHIPS": "Brasil — Blue Chips",
        "BR_SMALL_CAPS": "Brasil — Small Caps",
        "US_STOCKS": "EUA (Ações US)",
        "US_BLUE_CHIPS": "EUA — Blue Chips",
        "US_SMALL_CAPS": "EUA — Small Caps",
        "CRYPTO": "Criptos",
    }
    return mapping.get(name_key, name_key)


def _load_watchlists() -> Dict[str, List[str]]:
    # tenta usar a infra do projeto
    try:
        from core.data import load_watchlists as _lw  # type: ignore
        return _lw()
    except Exception:
        # fallback mínimo
        return {
            "BR_STOCKS": ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "ABEV3.SA"],
            "US_STOCKS": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"],
            "CRYPTO": ["BTC-USD", "ETH-USD", "SOL-USD"],
        }


# ======================================================================================
# Página
# ======================================================================================

st.set_page_config(page_title="Screener", page_icon="🧪", layout="wide")

st.title("Screener")
st.caption("Triagem multi-ativos (BR/US/Cripto) com métricas e filtros")

wls = _load_watchlists()
class_options = { _friendly(k): k for k in wls.keys() }  # "Brasil (Ações B3)" -> "BR_STOCKS"

with st.sidebar:
    st.header("Classe")
    class_label = st.selectbox("Classe", list(class_options.keys()))
    wl_key = class_options[class_label]
    symbols = wls.get(wl_key, [])

    st.caption(f"Total na classe: **{len(symbols)}**")

    st.header("Janela")
    period = st.selectbox("Período", ["6mo", "1y", "2y", "5y"], index=0)
    interval = st.selectbox("Intervalo", ["1d", "1wk", "1mo"], index=0)

    st.header("Filtros")
    min_price = st.number_input("Preço mínimo", min_value=0.0, value=1.0, step=0.5, format="%.2f")
    min_avgvol = st.number_input("Volume médio mínimo (unid.)", min_value=0.0, value=100_000.0, step=50_000.0, format="%.0f")
    only_trend_up = st.checkbox("Somente tendência de alta (SMA50>SMA200)", value=False)

    st.header("Limites")
    max_items = st.slider("Máx. de ativos processados", 5, 120, min(60, len(symbols) or 5), step=5)

st.write(f"Processando **{min(len(symbols), max_items)}** ativos desta classe…")

# =====================================================================
# Download
# =====================================================================

symbols = symbols[:max_items]
ver = int(datetime.utcnow().timestamp() // (60*30))  # muda versão a cada 30min p/ invalidar cache se quiser
data_dict = download_bulk(symbols, period, interval, ver=ver)

# =====================================================================
# Cálculo das métricas
# =====================================================================

rows = []
for s in symbols:
    df = data_dict.get(s)
    if df is None or df.empty:
        continue

    # Padroniza colunas
    df = df.copy()
    df.columns = [_flatten_col(c).title() for c in df.columns]

    # Fecha e volume robustos
    close_raw = df.get("Adj Close", df.get("Close"))
    vol_raw = df.get("Volume")

    close = _to_series(close_raw).dropna()
    vol = _to_series(vol_raw)

    last_price = float(close.iloc[-1]) if not close.empty else np.nan

    vol_tail = vol.tail(30).dropna()
    avg_vol = float(vol_tail.mean()) if not vol_tail.empty else np.nan

    # Retornos percentuais
    d1 = _ret(close, 1)
    d5 = _ret(close, 5)
    m1 = _ret(close, 21)
    m6 = _ret(close, 126)
    y1 = _ret(close, 252)

    rsi14 = _rsi(close, 14)
    volann = _vol_ann(close)

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    trend_up = (sma50 > sma200) if (np.isfinite(sma50) and np.isfinite(sma200)) else False

    # Score simples: média dos retornos normalizados + bônus de tendência
    comps = [d1, d5, m1, m6, y1]
    vals = [x for x in comps if pd.notna(x)]
    if vals:
        # normaliza p/ escala comparável
        score = float(np.nanmean([np.tanh(v * 3.0) for v in vals])) * 2.0  # -2..+2 aprox
    else:
        score = np.nan
    if trend_up and pd.notna(score):
        score += 0.5

    # Filtros
    if pd.notna(last_price) and last_price < min_price:
        continue
    if pd.notna(avg_vol) and avg_vol < min_avgvol:
        continue
    if only_trend_up and not trend_up:
        continue

    rows.append(
        {
            "Symbol": s,
            "Price": last_price,
            "D1%": d1,
            "D5%": d5,
            "M1%": m1,
            "M6%": m6,
            "Y1%": y1,
            "VolAnn%": volann / 100.0 if pd.notna(volann) else np.nan,  # guardo em fração para formatação %
            "AvgVol": avg_vol,
            "RSI14": rsi14,
            "TrendUp": trend_up,
            "Score": score,
        }
    )

df_res = pd.DataFrame(rows)

if df_res.empty:
    st.warning("Nenhum ativo passou pelos filtros. Ajuste os critérios.")
    st.stop()

# Ordenação
order_by = st.selectbox(
    "Ordenar por",
    ["Score", "M6%", "M1%", "Y1%", "D5%", "D1%", "RSI14", "VolAnn%", "AvgVol", "Price", "Symbol"],
    index=0,
)
ascending = st.checkbox("Ordem crescente", value=False)

# Para facilitar, converto nomes se usuário escolheu label que não é idêntico
sort_col_map = {
    "Score": "Score",
    "M6%": "M6%",
    "M1%": "M1%",
    "Y1%": "Y1%",
    "D5%": "D5%",
    "D1%": "D1%",
    "RSI14": "RSI14",
    "VolAnn%": "VolAnn%",
    "AvgVol": "AvgVol",
    "Price": "Price",
    "Symbol": "Symbol",
}
sort_col = sort_col_map.get(order_by, "Score")

df_res = df_res.sort_values(by=sort_col, ascending=ascending, kind="stable").reset_index(drop=True)

# ======================================================================================
# Tabela estilizada
# ======================================================================================

styled = (
    df_res.rename(
        columns={
            "Symbol": "Symbol",
            "Price": "Price",
            "D1%": "D1%",
            "D5%": "D5%",
            "M1%": "M1%",
            "M6%": "M6%",
            "Y1%": "Y1%",
            "VolAnn%": "VolAnn%",
            "AvgVol": "AvgVol",
            "RSI14": "RSI14",
            "TrendUp": "TrendUp",
            "Score": "Score",
        }
    )
    .style  # Pylance acusa "pd.Styler" mas é isso mesmo
    .map(_color_pct, subset=["D1%", "D5%", "M1%", "M6%", "Y1%"])
    .map(_color_score, subset=["Score"])
    .format(
        {
            "Price": _fmt_price,
            "D1%": _fmt_pct,
            "D5%": _fmt_pct,
            "M1%": _fmt_pct,
            "M6%": _fmt_pct,
            "Y1%": _fmt_pct,
            "VolAnn%": _fmt_pct,
            "AvgVol": _fmt_int,
            "RSI14": lambda x: "" if pd.isna(x) else f"{x:.1f}",
            "Score": lambda x: "" if pd.isna(x) else f"{x:.2f}",
        }
    )
)

st.dataframe(styled, height=520, use_container_width=True)

# ======================================================================================
# Enviar seleção ao Backtest (opcional)
# ======================================================================================

with st.expander("Marque os ativos que deseja enviar para o Backtest"):
    st.caption(
        "Você pode marcar símbolos manualmente e depois clicar no botão para salvá-los "
        "em `st.session_state['screener_selected']`."
    )

    # data_editor com checkbox
    edit_df = df_res[["Symbol", "Price", "Score"]].copy()
    edit_df.insert(0, "Select", False)
    edited = st.data_editor(
        edit_df,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Select": st.column_config.CheckboxColumn("Select"),
        },
        disabled=["Symbol", "Price", "Score"],
        height=280,
    )

    selected_syms = [row.Symbol for _, row in edited.iterrows() if row.Select]
    if st.button("Usar seleção no Backtest"):
        st.session_state["screener_selected"] = selected_syms
        st.success(f"{len(selected_syms)} ativo(s) enviado(s) para o Backtest.")


st.caption("Tip: limpe os caches em **Settings** se você notar dados antigos ou quiser forçar redownload.")
