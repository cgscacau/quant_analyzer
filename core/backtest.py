# core/backtest.py
from __future__ import annotations
import numpy as np
import pandas as pd

# --------- métricas ----------
def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty: return 0.0
    peak = equity.cummax()
    dd = equity/peak - 1.0
    return float(dd.min())  # negativo

def _annual_factor(freq: str) -> int:
    return 252 if str(freq).lower() in ("1d","d","day","daily") else 52

def kpis(returns: pd.Series, freq: str = "1d") -> dict:
    if returns is None or returns.empty:
        return dict(CAGR=0.0, Sharpe=0.0, MaxDD=0.0, Trades=0, WinRate=0.0)
    af = _annual_factor(freq)
    equity = (1 + returns).cumprod()
    cagr = equity.iloc[-1] ** (af / max(len(returns), 1)) - 1
    vol_ann = returns.std() * np.sqrt(af)
    sharpe = (returns.mean() * af) / (vol_ann + 1e-12)
    maxdd = _max_drawdown(equity)
    return dict(CAGR=float(cagr), Sharpe=float(sharpe), MaxDD=float(maxdd))

# --------- helpers ----------
def _apply_costs_and_alloc(strat_ret: pd.Series, signals: pd.Series, alloc: float, cost_bps_side: float) -> pd.Series:
    """
    Aplica alocação (0..1) e custos/slippage por LADO (entrada/saída).
    - signals: +1 entrada, -1 saída, 0 nada
    - custo por lado em bps (ex.: 5 bps = 0.0005)
    """
    alloc = float(max(0.0, min(1.0, alloc)))
    cost = (float(cost_bps_side) / 10000.0)  # bps -> fração
    # retorno bruto da estratégia considerando alocação
    adj = alloc * strat_ret
    # custo sempre que há troca de posição (entrada/saída)
    trans = signals.abs().fillna(0.0)
    adj = adj - (trans * cost)
    return adj

def _build_trade_log(close: pd.Series, signals: pd.Series, alloc: float, cost_bps_side: float) -> pd.DataFrame:
    """Log de trades; inclui Ret% bruto e líquido (após 2 lados de custo e alocação)."""
    trades = []
    in_trade = False
    entry_idx = None
    pos_map = {idx:i for i, idx in enumerate(close.index)}
    cost = float(cost_bps_side)/10000.0

    for idx, sig in signals.items():
        if sig == 1 and not in_trade:
            in_trade, entry_idx = True, idx
        elif sig == -1 and in_trade:
            entry_price, exit_price = float(close.loc[entry_idx]), float(close.loc[idx])
            ret_g = exit_price/entry_price - 1.0
            bars = pos_map[idx] - pos_map[entry_idx]
            # líquido (aprox): aplica alocação e desconta 2 lados de custo
            ret_n = (alloc * ret_g) - (2*cost)
            trades.append(dict(Entry=entry_idx, Exit=idx, Bars=int(bars),
                               EntryPx=entry_price, ExitPx=exit_price,
                               GrossPct=ret_g*100, NetPct=ret_n*100))
            in_trade, entry_idx = False, None

    if in_trade and entry_idx is not None:
        last_idx = close.index[-1]
        entry_price, exit_price = float(close.loc[entry_idx]), float(close.loc[last_idx])
        ret_g = exit_price/entry_price - 1.0
        bars = pos_map[last_idx] - pos_map[entry_idx]
        ret_n = (alloc * ret_g) - (2*cost)
        trades.append(dict(Entry=entry_idx, Exit=last_idx, Bars=int(bars),
                           EntryPx=entry_price, ExitPx=exit_price,
                           GrossPct=ret_g*100, NetPct=ret_n*100))

    df = pd.DataFrame(trades)
    if not df.empty:
        df = df[["Entry","Exit","Bars","EntryPx","ExitPx","GrossPct","NetPct"]]
    return df

# --------- estratégias ----------
def run_sma_cross(
    df: pd.DataFrame,
    fast: int = 20, slow: int = 50,
    freq: str = "1d",
    alloc: float = 1.0,
    fee_bps_side: float = 0.0,
    slippage_bps_side: float = 0.0,
) -> dict:
    close = df["Close"].dropna().copy()
    sma_fast = close.rolling(fast, min_periods=1).mean()
    sma_slow = close.rolling(slow, min_periods=1).mean()

    long = (sma_fast > sma_slow).astype(int)
    long_prev = long.shift(1).fillna(0)
    signals = long - long_prev  # +1 entrada, -1 saída

    # retornos do ativo
    ret = close.pct_change().fillna(0.0)
    strat_ret_raw = ret * long_prev  # posição aplicada na barra seguinte

    # custos + alocação
    cost_bps_side = (fee_bps_side + slippage_bps_side)
    strat_ret = _apply_costs_and_alloc(strat_ret_raw, signals, alloc, cost_bps_side)

    # benchmark (buy & hold com mesma alocação, sem custos)
    bench_ret = alloc * ret

    equity = (1 + strat_ret).cumprod()
    bench_eq = (1 + bench_ret).cumprod()

    trades_df = _build_trade_log(close, signals, alloc, cost_bps_side)
    trades = int(len(trades_df))
    wins = int((trades_df["NetPct"] > 0).sum()) if trades > 0 else 0

    m = kpis(strat_ret, freq=freq)
    m.update(dict(Trades=trades, WinRate=float(wins / trades * 100) if trades > 0 else 0.0))

    m_bh = kpis(bench_ret, freq=freq)

    return dict(
        returns=strat_ret, equity=equity, metrics=m, trades=trades_df,
        bench_returns=bench_ret, bench_equity=bench_eq, bench_metrics=m_bh
    )

def run_rsi_meanrev(
    df: pd.DataFrame,
    rsi_len: int = 14,
    buy_below: float = 30,
    sell_above: float = 70,  # reservado
    exit_mid: float = 50,
    freq: str = "1d",
    alloc: float = 1.0,
    fee_bps_side: float = 0.0,
    slippage_bps_side: float = 0.0,
) -> dict:
    close = df["Close"].dropna().copy()
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).ewm(alpha=1/rsi_len, adjust=False).mean()
    roll_dn = pd.Series(dn, index=close.index).ewm(alpha=1/rsi_len, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-12)
    rsi = 100 - (100 / (1 + rs))

    long = pd.Series(0, index=close.index, dtype=int)
    long = np.where(rsi < buy_below, 1, long)
    long = pd.Series(long, index=close.index)
    exit_sig = (rsi.shift(1) <= exit_mid) & (rsi > exit_mid)
    long = long.where(~exit_sig, 0).ffill().clip(0, 1)

    signals = pd.Series(long, index=close.index).diff().fillna(0)
    ret = close.pct_change().fillna(0.0)
    strat_ret_raw = ret * pd.Series(long, index=close.index).shift(1).fillna(0)

    cost_bps_side = (fee_bps_side + slippage_bps_side)
    strat_ret = _apply_costs_and_alloc(strat_ret_raw, signals, alloc, cost_bps_side)
    bench_ret = alloc * ret

    equity = (1 + strat_ret).cumprod()
    bench_eq = (1 + bench_ret).cumprod()

    trades_df = _build_trade_log(close, signals, alloc, cost_bps_side)
    trades = int(len(trades_df))
    wins = int((trades_df["NetPct"] > 0).sum()) if trades > 0 else 0

    m = kpis(strat_ret, freq=freq)
    m.update(dict(Trades=trades, WinRate=float(wins / trades * 100) if trades > 0 else 0.0))
    m_bh = kpis(bench_ret, freq=freq)

    return dict(
        returns=strat_ret, equity=equity, metrics=m, trades=trades_df,
        bench_returns=bench_ret, bench_equity=bench_eq, bench_metrics=m_bh
    )
