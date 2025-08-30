# core/portfolio_bt.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# ---------- helpers ----------
def _ann_factor(interval: str) -> int:
    return 252 if str(interval).lower() in ("1d","d","day","daily") else 52

def _period_end_mask(idx: pd.DatetimeIndex, kind: str) -> pd.Series:
    """
    Marca o último dia disponível de cada período.
    kind: 'none' | 'M' | 'Q' | 'A'
    """
    if kind == "none":
        m = pd.Series(False, index=idx)
        m.iloc[0] = True  # rebalance inicial
        return m

    # Define o agrupador por período
    if kind == "M":
        grp = idx.to_period("M")
    elif kind == "Q":
        grp = idx.to_period("Q")
    else:
        grp = idx.to_period("Y")

    # Converte o índice para Series para poder usar groupby/transform
    s = pd.Series(idx, index=idx)
    last_of_group = s.groupby(grp).transform("max")
    m = s.eq(last_of_group)
    m.iloc[0] = True
    return m

def _stats_from_equity(equity: pd.Series, interval: str) -> Dict[str,float]:
    if equity.empty:
        return dict(cagr=np.nan, vol=np.nan, sharpe=np.nan, maxdd=np.nan, total=np.nan)
    af = _ann_factor(interval)
    ret = equity.pct_change().dropna()
    vol = float(ret.std() * np.sqrt(af))
    ret_ann = float(ret.mean() * af)
    sharpe = ret_ann / (vol + 1e-12)
    total = float(equity.iloc[-1]/equity.iloc[0] - 1)
    peak = equity.cummax()
    dd = equity/peak - 1.0
    maxdd = float(dd.min())
    cagr = float((equity.iloc[-1]/equity.iloc[0])**(af/len(ret)) - 1) if len(ret)>0 else np.nan
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, maxdd=maxdd, total=total)

# ---------- backtest ----------
def backtest_portfolio(
    prices: pd.DataFrame,          # colunas = ativos (Close), index=dates
    weights: np.ndarray,           # soma 1
    interval: str = "1d",
    rebalance: str = "M",          # 'none'|'M'|'Q'|'A'
    init_cap: float = 100_000.0,
    mgmt_fee_annual: float = 0.0,  # % a.a. cobrada pró-rata por passo
    tc_bps: float = 0.0,           # custo de transação (bps) aplicado nos rebalances
    contrib_monthly: float = 0.0,  # aporte mensal (R$) aplicado no rebalance mensal
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Buy&Hold com rebalanceamento periódico para os 'weights' alvo.
    - Custos de transação aproximados por: turnover_weight * tc_bps.
    - Taxa de administração aplicada pró-rata por passo.
    - Aporte mensal rateado: soma no rebalance mensal (ou ignora se rebalance != M).
    Retorna (equity_series, df_info)
    """
    if prices.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    idx = prices.index
    ret = prices.pct_change().fillna(0.0)  # retornos simples
    af = _ann_factor(interval)
    fee_step = float(mgmt_fee_annual)/100.0/af
    tc_rate = float(tc_bps)/10_000.0

    # máscaras de rebalance
    kind_map = {
        "none": "none",
        "Mensal": "M", "Monthly": "M",
        "Trimestral": "Q", "Quarterly": "Q",
        "Anual": "A", "Annual": "A",
    }
    kind = kind_map.get(rebalance, "M")  # default para Mensal se vier algo inesperado

    mask_reb = _period_end_mask(idx, kind)

    # estado
    n = prices.shape[1]
    w_target = np.array(weights, dtype=float)
    w_target = w_target / w_target.sum()
    w_cur = w_target.copy()
    equity = np.empty(len(idx), dtype=float)
    equity[0] = init_cap

    # loop
    for t in range(1, len(idx)):
        # taxa de administração ao início do passo
        if fee_step:
            equity[t-1] *= (1.0 - fee_step)

        # retorno do dia
        r = ret.iloc[t].values  # (n,)
        gross = float(np.dot(w_cur, r))
        equity[t] = equity[t-1] * (1.0 + gross)

        # drift dos pesos após o retorno
        w_cur = w_cur * (1.0 + r)
        s = w_cur.sum()
        if s == 0:  # degenerate
            w_cur = w_target.copy()
        else:
            w_cur = w_cur / s

        # rebalance no fim do dia
        if bool(mask_reb.iloc[t]):
            # aporte mensal se rebalance mensal
            if rebalance in ("M","Monthly") and contrib_monthly>0:
                equity[t] += contrib_monthly
            # custo de transação proporcional à mudança de pesos
            turnover = np.abs(w_target - w_cur).sum() / 2.0   # fração do PL
            if tc_rate>0 and turnover>0:
                equity[t] *= (1.0 - tc_rate*turnover)
            w_cur = w_target.copy()

    eq = pd.Series(equity, index=idx, name="equity")
    info = pd.DataFrame({"ret": ret.dot(w_target)}, index=idx)
    return eq, info

# ---------- benchmark simples ----------
def bench_equity(bench_close: pd.Series, init_cap: float = 100_000.0) -> pd.Series:
    if bench_close is None or bench_close.empty:
        return pd.Series(dtype=float)
    eq = init_cap * (bench_close/bench_close.iloc[0])
    eq.name = "benchmark"
    return eq

def summarize_equities(port: pd.Series, bench: pd.Series | None, interval: str) -> Dict[str, Dict[str,float]]:
    out = {"portfolio": _stats_from_equity(port, interval)}
    if bench is not None and not bench.empty:
        out["benchmark"] = _stats_from_equity(bench, interval)
    return out
