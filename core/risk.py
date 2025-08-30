# core/risk.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ---------- métricas básicas ----------
def max_drawdown(equity: pd.Series) -> float:
    if equity is None or equity.empty: 
        return 0.0
    peak = equity.cummax()
    dd = equity/peak - 1.0
    return float(dd.min())  # negativo

def ann_vol(returns: pd.Series, freq: str = "1d") -> float:
    if returns is None or returns.empty: 
        return float("nan")
    scale = 252 if str(freq).lower() in ("1d","d") else 52
    return float(returns.std() * np.sqrt(scale))

# ---------- VaR / ES (histórico) ----------
def var_es_hist(returns: pd.Series, alpha: float = 0.95, horizon_days: int = 1) -> tuple[float,float]:
    """
    VaR/ES históricos de 1-período e escalados por sqrt(horizon) para múltiplos dias.
    Retornam valores negativos (perda), ex.: -0.031 = -3,1%.
    """
    r = returns.dropna()
    if r.empty:
        return float("nan"), float("nan")
    q = np.nanpercentile(r, (1 - alpha) * 100.0)  # quantil de perda
    es = r[r <= q].mean() if np.isfinite(q) else np.nan
    scale = np.sqrt(max(1, horizon_days))
    return float(q * scale), float(es * scale)

# ---------- ATR ----------
def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    h, l = df["High"], df["Low"]
    c_prev = df["Close"].shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - c_prev).abs(),
        (l - c_prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

# ---------- Sizing por risco ----------
def position_size(account_value: float, entry_price: float, stop_price: float, risk_pct: float = 1.0, lot_step: float = 1.0) -> dict:
    """
    Calcula quantidade para arriscar 'risk_pct'% do capital até o stop.
    lot_step: passo mínimo de lote (1 ação; cripto pode aceitar frações).
    """
    risk_capital = float(account_value) * (float(risk_pct)/100.0)
    risk_per_unit = max(1e-12, float(entry_price) - float(stop_price))
    qty_raw = risk_capital / risk_per_unit
    # arredonda para múltiplo do lote
    qty = np.floor(qty_raw / lot_step) * lot_step
    qty = max(qty, 0.0)
    notional = qty * float(entry_price)
    exp_loss = qty * (float(entry_price) - float(stop_price))
    return dict(qty=float(qty), notional=float(notional), expected_loss=float(exp_loss), risk_capital=risk_capital)

# ---------- Kelly ----------
def kelly_fraction(win_rate: float, payoff_ratio: float) -> float:
    """
    Kelly fracionado: f* = p - (1-p)/b, onde b = payoff_ratio = ganho médio / perda média.
    Retorna fração (0..1). Se inválido, devolve 0.
    """
    p = float(win_rate)/100.0
    b = float(payoff_ratio)
    if b <= 0 or p <= 0 or p >= 1:
        return 0.0
    f = p - (1 - p)/b
    return max(0.0, float(f))
