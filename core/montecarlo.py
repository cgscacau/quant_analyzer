# core/montecarlo.py
from __future__ import annotations
import numpy as np
import pandas as pd

# -------- utilidades --------
def ann_factor(interval: str) -> int:
    return 252 if str(interval).lower() in ("1d","d","day","daily") else 52

def _max_drawdown(equity: np.ndarray) -> float:
    """equity 1D array; retorna drawdown mínimo (negativo)."""
    peak = np.maximum.accumulate(equity)
    dd = equity/peak - 1.0
    return float(dd.min()) if dd.size else 0.0

# -------- simulações --------
def simulate_mvn_portfolio(
    rets_df: pd.DataFrame,
    weights: np.ndarray,
    steps: int,
    interval: str = "1d",
    n_sims: int = 5000,
    start_equity: float = 1.0,
    rebalance_every: int = 0,        # 0 = sem rebalance
    contrib_per_step: float = 0.0,   # aporte por passo (R$)
    mgmt_fee_annual: float = 0.0,    # % ao ano (ex.: 2.0)
    perf_fee_pct: float = 0.0,       # % sobre lucro acima do hurdle (ex.: 20.0)
    hurdle_annual: float = 0.0,      # % ao ano para hurdle (ex.: 6.0)
    withdraw_per_step: float = 0.0,  # saque por passo (R$)
) -> dict:
    n = len(rets_df.columns)
    af = ann_factor(interval)

    mu_ann = rets_df.mean().values * af
    cov_ann = rets_df.cov().values * af
    mu_step = mu_ann / af
    cov_step = cov_ann / af

    try:
        L = np.linalg.cholesky(cov_step)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(cov_step + np.eye(n)*1e-8)

    rng = np.random.default_rng(42)
    Z = rng.standard_normal(size=(steps, n_sims, n))
    shocks = np.tensordot(Z, L.T, axes=([2],[0]))
    R = shocks + mu_step.reshape(1,1,n)  # (T,S,n)

    equity = np.empty((steps+1, n_sims), dtype=float)
    equity[0,:] = start_equity

    w_target = weights.reshape(1, n)
    w_cur = np.repeat(w_target, n_sims, axis=0)

    mgmt_per_step   = float(mgmt_fee_annual)/100.0/af
    perf_fee_rate   = float(perf_fee_pct)/100.0
    hurdle_per_step = float(hurdle_annual)/100.0/af

    for t in range(steps):
        # aporte antes do retorno (mantém proporção)
        if contrib_per_step:
            equity[t,:] += contrib_per_step

        # taxa de administração (sobre PL)
        if mgmt_per_step:
            equity[t,:] *= (1.0 - mgmt_per_step)

        # rebalance?
        if rebalance_every and (t % rebalance_every == 0):
            w_cur[:] = w_target

        r_t = R[t, :, :]                     # (S,n)
        gross_ret = (w_cur * r_t).sum(axis=1)

        # taxa de performance sobre excedente ao hurdle do passo
        if perf_fee_rate:
            excess = np.maximum(gross_ret - hurdle_per_step, 0.0)
            perf_fee = perf_fee_rate * excess
            net_ret = gross_ret - perf_fee
        else:
            net_ret = gross_ret

        equity[t+1,:] = equity[t,:] * (1.0 + net_ret)

        # saque no fim do passo
        if withdraw_per_step:
            equity[t+1,:] = np.maximum(equity[t+1,:] - withdraw_per_step, 0.0)

        # drift de pesos
        w_cur = w_cur * (1.0 + r_t)
        w_cur = w_cur / w_cur.sum(axis=1, keepdims=True)

    return dict(equity=equity, r_port=None)



def simulate_bootstrap_portfolio(
    rets_df: pd.DataFrame,
    weights: np.ndarray,
    steps: int,
    n_sims: int = 5000,
    start_equity: float = 1.0,
    block: int = 1,
    rebalance_every: int = 0,        # 0 = sem rebalance
    contrib_per_step: float = 0.0,   # aporte por passo (R$)
    mgmt_fee_annual: float = 0.0,    # % ao ano
    perf_fee_pct: float = 0.0,       # % sobre lucro acima do hurdle
    hurdle_annual: float = 0.0,      # % ao ano
    withdraw_per_step: float = 0.0,  # saque por passo (R$)
) -> dict:
    X = rets_df.values
    T, n = X.shape
    af = ann_factor("1d")  # usamos o mesmo fator do modelo diário; para 1wk o passo é maior mas a conversão vem da página
    rng = np.random.default_rng(123)

    mgmt_per_step   = float(mgmt_fee_annual)/100.0/af
    perf_fee_rate   = float(perf_fee_pct)/100.0
    hurdle_per_step = float(hurdle_annual)/100.0/af

    equity = np.empty((steps+1, n_sims), dtype=float)
    equity[0,:] = start_equity

    for s in range(n_sims):
        w_target = weights.copy()
        w_cur = w_target.copy()
        e = start_equity
        t = 0
        while t < steps:
            i0 = rng.integers(0, max(1, T - block))
            Rblk = X[i0:i0+block]
            for r_t in Rblk:
                if t >= steps:
                    break

                if contrib_per_step:
                    e += contrib_per_step
                if mgmt_per_step:
                    e *= (1.0 - mgmt_per_step)
                if rebalance_every and (t % rebalance_every == 0):
                    w_cur = w_target.copy()

                gross_ret = float((w_cur * r_t).sum())
                if perf_fee_rate:
                    excess = max(gross_ret - hurdle_per_step, 0.0)
                    net_ret = gross_ret - perf_fee_rate*excess
                else:
                    net_ret = gross_ret

                e = e * (1.0 + net_ret)

                if withdraw_per_step:
                    e = max(e - withdraw_per_step, 0.0)

                w_cur = w_cur * (1.0 + r_t)
                ssum = max(1e-12, w_cur.sum())
                w_cur = w_cur / ssum
                t += 1
        # trajetória simplificada (preenche final e interpola linearmente)
        equity[1:, s] = np.linspace(start_equity, e, steps+1)[1:]
    return dict(equity=equity)


# -------- métricas/resultados --------
def fan_percentiles(equity: np.ndarray, percs=(5,25,50,75,95)) -> pd.DataFrame:
    """equity: (T+1, S) → DataFrame com percentis ao longo do tempo."""
    q = np.percentile(equity, percs, axis=1).transpose((1,0))  # (T+1, len(percs))
    cols = [f"p{p}" for p in percs]
    return pd.DataFrame(q, columns=cols)

def paths_summary(equity: np.ndarray, interval: str = "1d") -> dict:
    """Resumo sobre as S simulações (CAGR, MaxDD, terminal)."""
    T = equity.shape[0]-1
    af = ann_factor(interval)
    # retornos por barra
    rets = equity[1:,:]/equity[:-1,:] - 1.0                      # (T, S)
    # CAGR por sim
    cagr = (equity[-1,:] / equity[0,0]) ** (af / max(T,1)) - 1.0
    # MaxDD por sim
    maxdd = np.apply_along_axis(_max_drawdown, 0, equity)
    return dict(
        terminal=equity[-1,:].copy(),
        cagr=cagr.copy(),
        maxdd=maxdd.copy()
    )

def probability_targets(equity: np.ndarray, interval: str, target_cagr: float = 0.10, dd_thresh: float = -0.2) -> dict:
    """
    Probabilidade de terminar acima de uma meta anualizada e
    de sofrer drawdown abaixo de um limiar (negativo).
    """
    summ = paths_summary(equity, interval=interval)
    p_hit = float((summ["cagr"] >= target_cagr).mean()) if summ["cagr"].size else np.nan
    p_dd  = float((summ["maxdd"] <= dd_thresh).mean()) if summ["maxdd"].size else np.nan
    return dict(p_target=p_hit, p_dd=p_dd)
