# core/portfolio.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from core.risk import max_drawdown

# ----------------- utilidades -----------------
def ann_factor(interval: str) -> int:
    return 252 if str(interval).lower() in ("1d","d","day","daily") else 52

def ann_stats(ret_series: pd.Series, interval: str, rf_annual: float = 0.0) -> Dict[str, float]:
    """Estatísticas anuais para uma série de retornos simples (ex.: 0.01=1%)."""
    if ret_series is None or ret_series.empty:
        return dict(ret_ann=np.nan, vol_ann=np.nan, sharpe=np.nan, cagr=np.nan, maxdd=np.nan)
    af = ann_factor(interval)
    ret_ann = ret_series.mean() * af
    vol_ann = ret_series.std() * np.sqrt(af)
    sharpe = (ret_ann - rf_annual) / (vol_ann + 1e-12)
    equity = (1 + ret_series).cumprod()
    cagr = equity.iloc[-1] ** (af / max(len(ret_series), 1)) - 1
    mdd = max_drawdown(equity)
    return dict(ret_ann=float(ret_ann), vol_ann=float(vol_ann), sharpe=float(sharpe),
                cagr=float(cagr), maxdd=float(mdd))

def build_equal_weight_returns(rets_df: pd.DataFrame) -> pd.Series:
    n = rets_df.shape[1]
    if n == 0: return pd.Series(dtype=float)
    w = np.ones(n) / n
    return (rets_df * w).sum(axis=1)

# ----------------- simulações de portfólio -----------------
def random_weights(n_assets: int, w_max: float = 1.0, seed: int | None = 42) -> np.ndarray:
    """
    Gera pesos >=0 que somam 1, respeitando um teto por ativo (w_max). Heurística simples:
    amostra Dirichlet e, se estourar teto, redistribui o excesso.
    """
    rng = np.random.default_rng(seed)
    for _ in range(5000):
        w = rng.dirichlet(np.ones(n_assets))
        if (w <= w_max + 1e-12).all():
            return w
        # redistribuição: clip e renormaliza
        w = np.clip(w, 0, w_max)
        if w.sum() == 0:  # degenerate
            continue
        w = w / w.sum()
        if (w <= w_max + 1e-12).all():
            return w
    # fallback
    w = np.ones(n_assets) / n_assets
    return w

def mc_frontier(
    rets_df: pd.DataFrame,
    interval: str = "1d",
    rf_annual: float = 0.0,
    n_sims: int = 10000,
    w_max: float = 0.35,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Fronteira eficiente aproximada via Monte Carlo (sem short), versão VETORIZADA.
    - Gera todos os pesos de uma vez (Dirichlet)
    - Aplica teto por ativo (w_max) com clip + renormalização por linha
    - Calcula ret_ann/vol_ann/Sharpe com álgebra matricial (einsum)
    """
    names = rets_df.columns.tolist()
    n = len(names)
    rng = np.random.default_rng(seed)
    af = ann_factor(interval)

    # Estatísticas anuais por ativo
    mu = (rets_df.mean() * af).values          # (n,)
    cov = (rets_df.cov() * af).values          # (n,n)

    # Pesos aleatórios
    W = rng.dirichlet(np.ones(n), size=n_sims)  # (k,n)
    if w_max < 1.0:
        W = np.clip(W, 0.0, float(w_max))
        row_sums = W.sum(axis=1, keepdims=True)
        # se alguma linha zerou (tudo clipado), usa uniforme
        zero = (row_sums == 0.0).flatten()
        if zero.any():
            W[zero] = 1.0 / n
            row_sums = W.sum(axis=1, keepdims=True)
        W = W / row_sums

    # Retorno/vol/Sharpe vetorizados
    ret_ann = W @ mu                                     # (k,)
    vol_ann = np.sqrt(np.einsum('ij,jk,ik->i', W, cov, W))  # (k,)
    sharpe  = (ret_ann - rf_annual) / (vol_ann + 1e-12)

    df = pd.DataFrame({
        "ret_ann": ret_ann,
        "vol_ann": vol_ann,
        "sharpe":  sharpe,
    })
    for j, nm in enumerate(names):
        df[f"w_{nm}"] = W[:, j]
    return df


def extract_weights(row: pd.Series, names: List[str]) -> pd.Series:
    return pd.Series([row[f"w_{nm}"] for nm in names], index=names, dtype=float)
