# pages/5_🤖_ML_Models.py
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# Config da página
# ------------------------------------------------------------
st.set_page_config(page_title="ML Models", page_icon="🤖", layout="wide")


# ------------------------------------------------------------
# Watchlists (fallback seguro)
# ------------------------------------------------------------
def _load_watchlists_safe() -> Dict[str, List[str]]:
    try:
        from core.data import load_watchlists as _lw  # opcional, se existir
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
# Download robusto (com cache)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=900)
def download_history_batch(
    tickers: List[str], period: str, interval: str, min_rows: int = 100
) -> Dict[str, pd.DataFrame]:
    """Baixa candles via yfinance e retorna apenas os ativos válidos."""
    if not tickers:
        return {}

    tickers = list(dict.fromkeys([t.strip() for t in tickers if t and isinstance(t, str)]))

    # 1 ticker
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

    # Muitos tickers (MultiIndex)
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


def _choose_close_col(df: pd.DataFrame) -> pd.Series:
    if "Adj Close" in df.columns:
        return df["Adj Close"]
    if "Close" in df.columns:
        return df["Close"]
    return pd.Series(dtype=float, index=df.index)


# ------------------------------------------------------------
# Features & Target
# ------------------------------------------------------------
def rsi_series(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, macd_signal


def build_dataset(
    df: pd.DataFrame,
    rsi_len: int,
    sma_fast: int,
    sma_slow: int,
    vol_win: int,
    n_lags: int,
) -> pd.DataFrame:
    """Monta features e alvo (y = próxima barra de alta)."""
    close = _choose_close_col(df)
    if close.empty:
        return pd.DataFrame()

    ret = close.pct_change()

    # Features básicas
    feat = pd.DataFrame(index=close.index)
    feat["ret"] = ret
    feat["rsi"] = rsi_series(close, rsi_len)
    feat["sma_fast"] = close.rolling(sma_fast).mean()
    feat["sma_slow"] = close.rolling(sma_slow).mean()
    feat["sma_cross_strength"] = (feat["sma_fast"] - feat["sma_slow"]) / (feat["sma_slow"] + 1e-9)
    feat["vol"] = ret.rolling(vol_win).std()

    # MACD
    macd_line, macd_sig = macd(close)
    feat["macd"] = macd_line
    feat["macd_sig"] = macd_sig
    feat["macd_hist"] = macd_line - macd_sig

    # Lags de retorno
    for k in range(1, n_lags + 1):
        feat[f"ret_lag{k}"] = ret.shift(k)

    # Alvo: próxima barra sobe?
    feat["ret_next"] = ret.shift(-1)
    feat["y"] = (feat["ret_next"] > 0).astype(int)

    feat = feat.dropna()
    return feat


# ------------------------------------------------------------
# Treino e avaliação (TimeSeriesSplit OOF)
# ------------------------------------------------------------
def fit_oof_predict_proba(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    rf_n_estimators: int,
    rf_max_depth: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Faz OOF com TimeSeriesSplit (walk-forward) e retorna proba prevista (classe 1) e métricas médias.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    proba_oof = np.full(shape=len(X), fill_value=np.nan, dtype=float)

    # Modelos
    if model_type == "Logistic":
        base = LogisticRegression(max_iter=2000)
        scaler = StandardScaler()
        needs_scale = True
    elif model_type == "RandomForest":
        base = RandomForestClassifier(
            n_estimators=rf_n_estimators, max_depth=None if rf_max_depth <= 0 else rf_max_depth,
            random_state=7, n_jobs=-1
        )
        needs_scale = False
        scaler = None
    elif model_type == "HistGB":
        base = HistGradientBoostingClassifier(random_state=7)
        needs_scale = False
        scaler = None
    else:  # Stacking (RF + HGB + LR)
        rf = RandomForestClassifier(n_estimators=rf_n_estimators, random_state=7, n_jobs=-1)
        hgb = HistGradientBoostingClassifier(random_state=7)
        lr = LogisticRegression(max_iter=2000)
        estimators = [("rf", rf), ("hgb", hgb)]
        base = StackingClassifier(estimators=estimators, final_estimator=lr, passthrough=True, n_jobs=-1)
        needs_scale = False
        scaler = None

    # OOF
    for tr_idx, te_idx in tscv.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        if needs_scale:
            scaler.fit(X_tr)
            X_tr2 = pd.DataFrame(scaler.transform(X_tr), index=X_tr.index, columns=X_tr.columns)
            X_te2 = pd.DataFrame(scaler.transform(X_te), index=X_te.index, columns=X_te.columns)
        else:
            X_tr2, X_te2 = X_tr, X_te

        base.fit(X_tr2, y_tr)
        if hasattr(base, "predict_proba"):
            p = base.predict_proba(X_te2)[:, 1]
        else:
            # modelos sem predict_proba
            p = base.predict(X_te2).astype(float)

        proba_oof[te_idx] = p

    # Métricas em OOF (descarta NaN do início)
    mask = ~np.isnan(proba_oof)
    y_valid = y.iloc[mask]
    p_valid = proba_oof[mask]

    preds = (p_valid >= 0.5).astype(int)
    metrics = {
        "Accuracy": float(accuracy_score(y_valid, preds)),
        "Precision": float(precision_score(y_valid, preds, zero_division=0)),
        "Recall": float(recall_score(y_valid, preds, zero_division=0)),
        "F1": float(f1_score(y_valid, preds, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_valid, p_valid)) if len(np.unique(y_valid)) > 1 else 0.5,
    }
    return proba_oof, metrics


def strategy_from_proba(
    proba: np.ndarray,
    ret_next: pd.Series,
    buy_threshold: float,
    cost_bps_side: float = 0.0,
) -> pd.Series:
    """
    Gera equity usando proba OOF (compra em t se proba(t)≥th e realiza em t+1).
    Custo por mudança de posição (bps) aplicado em variações 0->1 e 1->0.
    """
    if ret_next.empty:
        return pd.Series([], dtype=float)

    p = pd.Series(proba, index=ret_next.index)
    p = p.dropna()
    retn = ret_next.loc[p.index]

    pos = (p >= buy_threshold).astype(int)
    # retorno com posição (entrada no próximo candle é aproximada com OOF)
    strat_ret = retn * pos

    # custos
    changes = pos.diff().abs().fillna(0.0)
    bps_cost = cost_bps_side / 10000.0
    strat_ret = strat_ret - changes * bps_cost

    equity = (1.0 + strat_ret.fillna(0.0)).cumprod()
    return equity


def _metrics_from_equity(equity: pd.Series, ret_freq: str = "D") -> Tuple[float, float, float]:
    if equity.empty or len(equity) < 2:
        return 0.0, 0.0, 0.0

    total_ret = equity.iloc[-1]
    days = (equity.index[-1] - equity.index[0]).days or 1
    years = days / 365.25
    cagr = (total_ret ** (1 / years) - 1) if total_ret > 0 and years > 0 else 0.0

    rets = equity.pct_change().dropna()
    if rets.empty:
        sharpe = 0.0
    else:
        ann = 252 if ret_freq.upper().startswith("D") else 52
        sharpe = (rets.mean() / (rets.std() + 1e-9)) * math.sqrt(ann)

    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    max_dd = dd.min() if len(dd) else 0.0

    return float(cagr), float(sharpe), float(abs(max_dd))


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("🤖 ML Models")
st.caption("Classificação de próxima barra (subida/queda) com Logistic / RF / HistGB / Stacking")

watch = _load_watchlists_safe()
universe = _flatten_unique(watch.values())
screener_sel = st.session_state.get("screener_selected", [])

use_screener = st.toggle(
    "Usar seleção do Screener (se houver)",
    value=bool(screener_sel),
)
symbols_manual = st.multiselect(
    "Ativos para treinar",
    options=universe,
    default=(screener_sel or ["AAPL", "MSFT", "NVDA", "PETR4.SA"]),
    disabled=use_screener,
)
symbols = screener_sel if (use_screener and screener_sel) else symbols_manual

cols_top = st.columns(3)
with cols_top[0]:
    period = st.selectbox("Período", ["6mo", "1y", "2y", "5y"], index=1)
with cols_top[1]:
    interval = st.selectbox("Intervalo", ["1d", "1h", "1wk"], index=0)
with cols_top[2]:
    dark_theme = st.toggle("Tema escuro", value=False)

st.markdown("### Features")
f1, f2, f3, f4, f5 = st.columns(5)
with f1:
    rsi_len = st.slider("RSI length", 5, 40, 14)
with f2:
    sma_fast = st.slider("SMA rápida", 5, 100, 10)
with f3:
    sma_slow = st.slider("SMA lenta", 10, 200, 30)
with f4:
    vol_win = st.slider("Janela de volatilidade", 5, 60, 20)
with f5:
    n_lags = st.slider("Lags de retorno", 0, 10, 5)

st.markdown("### Modelo")
mcol1, mcol2, mcol3 = st.columns(3)
with mcol1:
    model_type = st.radio("Tipo", ["RandomForest", "HistGB", "Logistic", "Stacking"], horizontal=False, index=0)
with mcol2:
    buy_threshold = st.slider("Limiar de compra (prob. de alta)", 0.5, 0.9, 0.55, step=0.01)
with mcol3:
    rf_n_estimators = st.slider("n_estimators (RF)", 50, 1000, 300, step=25)
    rf_max_depth = st.slider("max_depth (RF) (0=auto)", 0, 20, 0)

cst1, cst2 = st.columns(2)
with cst1:
    cost_bps_side = st.number_input("Custo por lado (bps)", min_value=0.0, value=0.0, step=0.25)
with cst2:
    show_prob_plot = st.toggle("Mostrar gráfico de probabilidade", value=True)

# ------------------------------------------------------------
# Download
# ------------------------------------------------------------
if not symbols:
    st.info("Nenhum ativo selecionado. Selecione no Screener e depois ative o toggle acima, ou escolha manualmente.")
    st.stop()

with st.spinner("Baixando dados..."):
    bars_map = download_history_batch(symbols, period, interval, min_rows=150)

mantidos = list(bars_map.keys())
descartados = [s for s in symbols if s not in mantidos]
st.caption(f"✓ Válidos: {len(mantidos)} | ✗ Sem dados: {len(descartados)}")
if descartados:
    st.caption(f"Descartados: {', '.join(descartados)}")

if not mantidos:
    st.warning("Nenhum ativo com dados válidos. Tente outro período/intervalo.")
    st.stop()

# ------------------------------------------------------------
# Treino / Avaliação por ativo
# ------------------------------------------------------------
all_metrics = {}
equities = {}
prob_series_map = {}

for sym, df in bars_map.items():
    # Garante índice de datas
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception:
            continue

    ds = build_dataset(df, rsi_len, sma_fast, sma_slow, vol_win, n_lags)
    if ds.empty or ds.shape[0] < 150:
        continue

    features = [c for c in ds.columns if c not in ("y", "ret_next")]
    X = ds[features]
    y = ds["y"].astype(int)

    proba_oof, m = fit_oof_predict_proba(
        X=X, y=y, model_type=model_type,
        rf_n_estimators=rf_n_estimators, rf_max_depth=rf_max_depth
    )
    all_metrics[sym] = m

    # Estratégia
    equity = strategy_from_proba(
        proba=proba_oof, ret_next=ds["ret_next"], buy_threshold=buy_threshold, cost_bps_side=cost_bps_side
    )
    equities[sym] = equity

    # Probabilidade para plot
    prob_series_map[sym] = pd.Series(proba_oof, index=ds.index)

if not all_metrics:
    st.warning("Não foi possível treinar/avaliar (dados insuficientes após filtros).")
    st.stop()

# ------------------------------------------------------------
# Tabela de métricas (modelo) e desempenho (estratégia)
# ------------------------------------------------------------
met_df = pd.DataFrame(all_metrics).T[
    ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
].sort_values(by=["ROC_AUC", "F1"], ascending=False)
met_df = met_df.round(4)

# Estratégia
perf_rows = {}
for sym, eq in equities.items():
    cagr, sharpe, mdd = _metrics_from_equity(eq)
    perf_rows[sym] = {"CAGR%": 100 * cagr, "Sharpe": sharpe, "MaxDD%": 100 * mdd}
perf_df = pd.DataFrame(perf_rows).T[["CAGR%", "Sharpe", "MaxDD%"]].sort_values(by=["CAGR%", "Sharpe"], ascending=False)
perf_df = perf_df.round({"CAGR%": 2, "MaxDD%": 2, "Sharpe": 2})

st.subheader("Métricas do Modelo (OOF)")
st.dataframe(met_df, use_container_width=True, height=260)

st.subheader("Desempenho da Estratégia (prob ≥ limiar)")
st.dataframe(perf_df, use_container_width=True, height=240)

best = perf_df.index[0]
st.success(f"Melhor por CAGR: **{best}** (CAGR {perf_df.loc[best, 'CAGR%']:.2f}%, Sharpe {perf_df.loc[best, 'Sharpe']:.2f})")

# ------------------------------------------------------------
# Gráficos
# ------------------------------------------------------------
template = "plotly_dark" if dark_theme else "plotly"
tabs = st.tabs([f"{s}" for s in equities.keys()])

for tab, sym in zip(tabs, equities.keys()):
    with tab:
        eq = equities[sym]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Equity"))
        fig.update_layout(
            template=template, height=360, margin=dict(l=10, r=10, t=30, b=10),
            title=f"{sym} — Equity (ML strategy)", yaxis_title="Equity", xaxis_title="Tempo"
        )
        st.plotly_chart(fig, use_container_width=True)

        if show_prob_plot and sym in prob_series_map:
            p = prob_series_map[sym].dropna()
            thr = pd.Series(buy_threshold, index=p.index)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=p.index, y=p.values, name="P(up) OOF", mode="lines"))
            fig2.add_trace(go.Scatter(x=thr.index, y=thr.values, name="Limiar", mode="lines", line=dict(dash="dash")))
            fig2.update_layout(
                template=template, height=260, margin=dict(l=10, r=10, t=30, b=10),
                title=f"{sym} — Probabilidade prevista (OOF)", yaxis_title="P(up)", xaxis_title="Tempo"
            )
            st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# ✅ Resumo e plano tático (experimental)
# ============================================================
# === BLOCO: Resumo e plano tático (substitua o atual) =======================

with st.expander(f"📌 {sym} — Situação e plano"):
    # 1) Pegar a última probabilidade (se existir)
    if 'p_series' in locals() and p_series is not None and len(p_series) > 0:
        try:
            p_up = float(p_series.iloc[-1])
        except Exception:
            p_up = np.nan
    else:
        p_up = np.nan

    # 2) Preparar strings seguras (N/A quando NaN)
    p_up_txt  = "N/A" if np.isnan(p_up) else f"{p_up:.2%}"
    rsi_txt   = "N/A" if ('rsi_val' not in locals() or rsi_val is None or np.isnan(rsi_val)) else f"{float(rsi_val):.1f}"
    atr_txt   = "N/A" if ('atr_val' not in locals() or atr_val is None or np.isnan(atr_val) or atr_val <= 0) else f"{float(atr_val):.2f}"
    sw_hi_txt = "N/A" if ('sw_hi'   not in locals() or sw_hi   is None or np.isnan(sw_hi))   else f"{float(sw_hi):.2f}"
    sw_lo_txt = "N/A" if ('sw_lo'   not in locals() or sw_lo   is None or np.isnan(sw_lo))   else f"{float(sw_lo):.2f}"

    # 3) Status de tendência/macd com segurança
    trend_txt = "Alta" if ('fast_gt_slow' in locals() and fast_gt_slow) else "Baixa/Fraca"
    macd_txt  = "Positivo" if ('macd_pos' in locals() and macd_pos) else "Negativo"

    # 4) Cabeçalho da situação
    st.markdown(
        f"**Preço atual:** `{px:.2f}`  |  **Prob. modelo (alta):** `{p_up_txt}`  \n"
        f"**Tendência:** `{trend_txt}`  |  **MACD:** `{macd_txt}`  |  **RSI:** `{rsi_txt}`  \n"
        f"**Swing (20):** máx `{sw_hi_txt}` | mín `{sw_lo_txt}`  |  **ATR(14):** `{atr_txt}`"
    )

    # 5) Plano tático (entrada/stop/alvo) quando houver sinal
    go_long_flag = 'go_long' in locals() and go_long
    buy_th = float(buy_threshold) if 'buy_threshold' in locals() else 0.55

    if go_long_flag:
        # Valores defensivos
        stop_val = float(stop_lvl) if ('stop_lvl' in locals() and stop_lvl is not None and not np.isnan(stop_lvl)) else np.nan
        tgt_val  = float(target)   if ('target'   in locals() and target   is not None and not np.isnan(target))   else np.nan

        # Risco e R:R (com proteção contra divisão por zero/NaN)
        risk = (px - stop_val) if (not np.isnan(stop_val)) else np.nan
        if (risk is None) or np.isnan(risk) or risk <= 0:
            rr = np.nan
        else:
            rr = (tgt_val - px) / risk if (not np.isnan(tgt_val) and tgt_val > px) else np.nan

        stop_txt = "N/A" if np.isnan(stop_val) else f"{stop_val:.2f}"
        tgt_txt  = "N/A" if np.isnan(tgt_val)  else f"{tgt_val:.2f}"
        rr_txt   = "N/A" if np.isnan(rr)       else f"{rr:.2f}"

        st.success(
            f"**Sinal LONG (experimental)** — prob ≥ limiar ({buy_th:.2f}).\n"
            f"- **Entrada:** ~ **{px:.2f}**\n"
            f"- **Stop:** ~ **{stop_txt}**\n"
            f"- **Alvo:** ~ **{tgt_txt}**  (R:R ≈ {rr_txt})"
        )
    else:
        st.info(
            "Sem entrada **LONG** agora. "
            "Aguarde: prob acima do limiar, médias alinhadas (SMA rápida>lenta), MACD>0 e RSI entre 45–70."
        )

    # 6) Motivos/observações (se houver)
    if 'reasons' in locals() and reasons:
        st.caption("• Motivos: " + "; ".join(reasons))
    st.caption("※ Conteúdo educacional; não é recomendação de investimento.")
# ============================================================================

