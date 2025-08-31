# pages/4_📈_Backtest.py
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ------------------------------------------------------------
# Configuração da página
# ------------------------------------------------------------
st.set_page_config(page_title="Backtest", page_icon="📈", layout="wide")


# ------------------------------------------------------------
# Utilidades de watchlists (fallback seguro)
# ------------------------------------------------------------
def _load_watchlists_safe() -> Dict[str, List[str]]:
    """Tenta carregar watchlists do projeto; se falhar, usa um conjunto básico."""
    try:
        from core.data import load_watchlists as _lw  # opcional
        return _lw()
    except Exception:
        return {
            "BR_STOCKS": ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA"],
            "FIIs": ["CPTS11.SA", "RBVA11.SA", "XPML11.SA", "RBRF11.SA", "GGRC11.SA", "HGBS11.SA", "KNCR11.SA"],
            "US_STOCKS": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"],
            "CRYPTO": ["BTC-USD", "ETH-USD", "SOL-USD"],
        }


def _flatten_unique(lst_of_lists: List[List[str]]) -> List[str]:
    out, seen = [], set()
    for lst in lst_of_lists:
        for x in lst:
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out


# ------------------------------------------------------------
# Download robusto de histórico (com cache)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=900)
def download_history_batch(
    tickers: List[str], period: str, interval: str, min_rows: int = 30
) -> Dict[str, pd.DataFrame]:
    """
    Baixa candles via yfinance de vários tickers.
    Retorna dict {ticker: DataFrame OHLCV} com somente os válidos.
    """
    if not tickers:
        return {}

    # Normaliza para lista única (evita duplicados)
    tickers = list(dict.fromkeys([t.strip() for t in tickers if t and isinstance(t, str)]))

    # Um único ticker: yfinance retorna DF simples (colunas diretas)
    if len(tickers) == 1:
        df = yf.download(
            tickers[0], period=period, interval=interval,
            auto_adjust=False, threads=True, progress=False
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.dropna(how="all")
            ok_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume"} & set(map(str, df.columns))
            if len(df) >= min_rows and ok_cols:
                return {tickers[0]: df.copy()}
        return {}

    # Vários tickers -> yfinance retorna MultiIndex nas colunas: (ticker, campo)
    df = yf.download(
        tickers, period=period, interval=interval,
        group_by="ticker", auto_adjust=False, threads=True, progress=False
    )

    out: Dict[str, pd.DataFrame] = {}
    if isinstance(df, pd.DataFrame) and not df.empty:
        for t in tickers:
            try:
                sub = df[t].copy()
            except Exception:
                continue
            if not isinstance(sub, pd.DataFrame) or sub.empty:
                continue
            sub = sub.dropna(how="all")
            ok_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume"} & set(map(str, sub.columns))
            if len(sub) >= min_rows and ok_cols:
                out[t] = sub
    return out


# ------------------------------------------------------------
# Indicadores e estratégias
# ------------------------------------------------------------
def _choose_close_col(df: pd.DataFrame) -> pd.Series:
    """Escolhe coluna de preço de fechamento (Adj Close, se existir)."""
    if "Adj Close" in df.columns:
        return df["Adj Close"]
    if "Close" in df.columns:
        return df["Close"]
    # fallback raro
    # se não houver nada conhecido, cria série vazia (invalida)
    return pd.Series(dtype=float, index=df.index)


def rsi_series(close: pd.Series, length: int = 14) -> pd.Series:
    """RSI clássico (Wilder) usando médias exponenciais."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing (EMA com alpha=1/length)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def trades_from_position(close: pd.Series, pos: pd.Series) -> Tuple[int, float]:
    """
    Computa número de trades e win-rate aproximado.
    Entrada/saída = pontos onde pos muda 0->1 ou 1->0. Win-rate baseado em retorno do trade.
    """
    pos = pos.fillna(0).astype(int)
    changes = pos.diff().fillna(0)

    entries = list(close.index[changes == 1])
    exits = list(close.index[changes == -1])

    # Ajusta pares (começar por entrada e terminar na saída)
    if len(exits) and len(entries):
        if exits[0] < entries[0]:
            exits = exits[1:]
        if len(entries) > len(exits):
            entries = entries[: len(exits)]

    wins = 0
    rets = []
    for e, x in zip(entries, exits):
        if close.loc[e] > 0 and close.loc[x] > 0:
            r = (close.loc[x] / close.loc[e]) - 1.0
            rets.append(r)
            if r > 0:
                wins += 1

    n_trades = len(rets)
    win_rate = (wins / n_trades) if n_trades else 0.0
    return n_trades, win_rate


def backtest_sma(
    df: pd.DataFrame,
    n_fast: int,
    n_slow: int,
    cost_bps_side: float,
    slip_bps: float,
    alloc_pct: float,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Estratégia Momentum (SMA Cross).
    Retorna equity normalizada (1.0 inicial) e dict de métricas.
    """
    close = _choose_close_col(df)
    if close.empty or len(close) < max(n_fast, n_slow) + 5:
        return pd.Series([], dtype=float), {}

    sma_f = close.rolling(n_fast).mean()
    sma_s = close.rolling(n_slow).mean()

    pos = (sma_f > sma_s).astype(int).shift(1).fillna(0)  # entra no próximo candle
    ret = close.pct_change().fillna(0.0)

    strat_ret = ret * pos

    # custos quando muda a posição
    changes = pos.diff().abs().fillna(0.0)
    bps_cost = (cost_bps_side + slip_bps) / 10000.0
    strat_ret = strat_ret - changes * bps_cost

    # alocação de capital
    strat_ret = strat_ret * (alloc_pct / 100.0)

    equity = (1.0 + strat_ret).cumprod()

    # métricas
    n_trades, win_rate = trades_from_position(close, pos)
    cagr, sharpe, mdd = _metrics_from_equity(equity, ret_freq="D")

    metrics = {
        "CAGR%": 100 * cagr,
        "Sharpe": sharpe,
        "MaxDD%": 100 * mdd,
        "Trades": n_trades,
        "WinRate%": 100 * win_rate,
    }
    return equity, metrics


def backtest_rsi(
    df: pd.DataFrame,
    rsi_len: int,
    buy_th: int,
    exit_th: int,
    cost_bps_side: float,
    slip_bps: float,
    alloc_pct: float,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Estratégia Mean Reversion via RSI (compra < buy_th, zera > exit_th).
    """
    close = _choose_close_col(df)
    if close.empty or len(close) < rsi_len + 5:
        return pd.Series([], dtype=float), {}

    rsi = rsi_series(close, rsi_len)

    # Gera série de posição (0/1) iterativa para respeitar buy/exit
    pos = pd.Series(0, index=close.index, dtype=int)
    holding = 0
    for i in range(1, len(pos)):
        if holding == 0:
            if rsi.iloc[i - 1] < buy_th:
                holding = 1
        else:
            if rsi.iloc[i - 1] > exit_th:
                holding = 0
        pos.iloc[i] = holding

    ret = close.pct_change().fillna(0.0)
    strat_ret = ret * pos.shift(0)  # posição válida já aplicada no candle atual (sem lookahead)

    # custos em mudanças
    changes = pos.diff().abs().fillna(0.0)
    bps_cost = (cost_bps_side + slip_bps) / 10000.0
    strat_ret = strat_ret - changes * bps_cost

    # alocação de capital
    strat_ret = strat_ret * (alloc_pct / 100.0)

    equity = (1.0 + strat_ret).cumprod()

    # métricas
    n_trades, win_rate = trades_from_position(close, pos)
    cagr, sharpe, mdd = _metrics_from_equity(equity, ret_freq="D")

    metrics = {
        "CAGR%": 100 * cagr,
        "Sharpe": sharpe,
        "MaxDD%": 100 * mdd,
        "Trades": n_trades,
        "WinRate%": 100 * win_rate,
    }
    return equity, metrics


def _metrics_from_equity(equity: pd.Series, ret_freq: str = "D") -> Tuple[float, float, float]:
    """
    Retorna (CAGR, Sharpe, MaxDrawdown) dados equity normalizada (1.0 no início).
    """
    if equity.empty or len(equity) < 2:
        return 0.0, 0.0, 0.0

    # CAGR
    total_ret = equity.iloc[-1]
    days = (equity.index[-1] - equity.index[0]).days or 1
    years = days / 365.25
    cagr = (total_ret ** (1 / years) - 1) if total_ret > 0 and years > 0 else 0.0

    # Sharpe
    rets = equity.pct_change().dropna()
    if rets.empty:
        sharpe = 0.0
    else:
        ann = 252 if ret_freq.upper().startswith("D") else 52
        sharpe = (rets.mean() / (rets.std() + 1e-9)) * math.sqrt(ann)

    # Max Drawdown
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    max_dd = dd.min() if len(dd) else 0.0

    return float(cagr), float(sharpe), float(abs(max_dd))


# ------------------------------------------------------------
# UI — Seleção, período/intervalo, estratégia, custos
# ------------------------------------------------------------
st.title("📈 Backtest")
st.caption("Comparação rápida de estratégias por ativo")

watch = _load_watchlists_safe()
universe = _flatten_unique(watch.values())
screener_sel = st.session_state.get("screener_selected", [])

use_screener = st.toggle(
    "Usar seleção do Screener (se houver)",
    value=bool(screener_sel),
    help="Se ligado e existir uma seleção salva pelo Screener, ela é usada. Caso contrário, use o campo abaixo.",
)

symbols_manual = st.multiselect(
    "Escolha os ativos (caso não use a seleção do Screener)",
    options=universe,
    default=(screener_sel or ["AAPL"]),
    disabled=use_screener,
)

symbols = screener_sel if (use_screener and screener_sel) else symbols_manual

with st.expander("Debug seleção", expanded=False):
    st.write("screener_selected:", screener_sel)
    st.write("use_screener:", use_screener)
    st.write("symbols (efetivos):", symbols)

cols_p = st.columns(3)
with cols_p[0]:
    period = st.selectbox("Período", ["6mo", "1y", "2y", "5y", "max"], index=1)
with cols_p[1]:
    interval = st.selectbox("Intervalo", ["1d", "1h", "1wk"], index=0)
with cols_p[2]:
    dark_theme = st.toggle("Tema escuro (gráficos)", value=False)

st.markdown("### Estratégia")
stype = st.radio("Tipo", ["Momentum (SMA Cross)", "Mean Reversion (RSI)"], horizontal=False, index=0)

if stype == "Momentum (SMA Cross)":
    c1, c2 = st.columns(2)
    with c1:
        sma_fast = st.slider("SMA Rápida", min_value=5, max_value=100, value=20, step=1)
    with c2:
        sma_slow = st.slider("SMA Lenta", min_value=10, max_value=200, value=50, step=1)
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        rsi_len = st.slider("RSI length", min_value=5, max_value=50, value=14, step=1)
    with c2:
        rsi_buy = st.slider("Compra se RSI <", min_value=5, max_value=50, value=30, step=1)
    with c3:
        rsi_exit = st.slider("Zera se RSI >", min_value=10, max_value=90, value=50, step=1)

st.markdown("### Custos, slippage & sizing")
cc1, cc2, cc3 = st.columns(3)
with cc1:
    cost_bps_side = st.number_input("Custo por lado (bps)", min_value=0.0, value=2.0, step=0.25, help="Cobrado a cada mudança de posição (entrada/saída).")
with cc2:
    slip_bps = st.number_input("Slippage por lado (bps)", min_value=0.0, value=3.0, step=0.25)
with cc3:
    alloc_pct = st.slider("Alocação do capital (%)", min_value=0, max_value=100, value=100, step=5)

# ------------------------------------------------------------
# Regras de validação e download
# ------------------------------------------------------------
if not symbols:
    st.info("Nenhum ativo selecionado. Selecione no Screener e clique em **Usar seleção no Backtest** (no Screener) ou escolha manualmente acima.")
    st.stop()

with st.spinner("Baixando dados..."):
    bars_map = download_history_batch(symbols, period, interval, min_rows=30)

mantidos = list(bars_map.keys())
descartados = [s for s in symbols if s not in mantidos]

st.caption(f"✓ Válidos: {len(mantidos)} | ✗ Sem dados: {len(descartados)}")
if descartados:
    st.caption(f"Descartados: {', '.join(descartados)}")

if not mantidos:
    st.warning("Nenhum ativo com dados válidos para este período/intervalo. Tente outro período/intervalo ou menos tickers.")
    st.stop()

# ------------------------------------------------------------
# Backtest por ativo
# ------------------------------------------------------------
results: Dict[str, Dict[str, float]] = {}
equities: Dict[str, pd.Series] = {}

for sym, df in bars_map.items():
    # Garante index tipo Datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception:
            continue

    if stype == "Momentum (SMA Cross)":
        eq, met = backtest_sma(
            df=df,
            n_fast=sma_fast,
            n_slow=sma_slow,
            cost_bps_side=cost_bps_side,
            slip_bps=slip_bps,
            alloc_pct=alloc_pct,
        )
    else:
        eq, met = backtest_rsi(
            df=df,
            rsi_len=rsi_len,
            buy_th=rsi_buy,
            exit_th=rsi_exit,
            cost_bps_side=cost_bps_side,
            slip_bps=slip_bps,
            alloc_pct=alloc_pct,
        )

    if not eq.empty and met:
        equities[sym] = eq
        results[sym] = met

if not results:
    st.warning("Nenhum resultado gerado (dados insuficientes após filtros/estratégia).")
    st.stop()

# ------------------------------------------------------------
# Tabela de métricas
# ------------------------------------------------------------
metrics_df = (
    pd.DataFrame(results)
    .T[["CAGR%", "Sharpe", "MaxDD%", "Trades", "WinRate%"]]
    .sort_values(by=["CAGR%","Sharpe"], ascending=[False, False])
)
metrics_df = metrics_df.round({"CAGR%": 2, "MaxDD%": 2, "WinRate%": 1, "Sharpe": 2})

st.subheader("Resumo (métricas por ativo)")
st.dataframe(metrics_df, use_container_width=True, height=340)

best = metrics_df.index[0]

# ------------------------------------------------------------
# Gráficos de equity
# ------------------------------------------------------------
st.subheader("Equity por ativo")

template = "plotly_dark" if dark_theme else "plotly"

tabs = st.tabs([f"{sym}" for sym in equities.keys()])
for tab, sym in zip(tabs, equities.keys()):
    with tab:
        eq = equities[sym]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Equity", mode="lines"))
        fig.update_layout(
            template=template,
            height=380,
            margin=dict(l=10, r=10, t=30, b=10),
            title=f"{sym} — Equity ({stype})",
            yaxis_title="Equity (R$ virtual)",
            xaxis_title="Tempo",
        )
        st.plotly_chart(fig, use_container_width=True)

st.success(f"Pronto! Melhor por CAGR no conjunto atual: **{best}** (CAGR {metrics_df.loc[best, 'CAGR%']:.2f}%, Sharpe {metrics_df.loc[best, 'Sharpe']:.2f})")
