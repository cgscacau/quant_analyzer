# core/ml.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ------------------- Indicadores / Features -------------------

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-12)
    return 100 - (100 / (1 + rs))

@dataclass
class FeatureConfig:
    rsi_len: int = 14
    sma_fast: int = 10
    sma_slow: int = 30
    n_lags: int = 5
    vol_win: int = 20

def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Retorna X, y, feature_names para um DF OHLCV com coluna Close."""
    c = df["Close"].astype(float)
    ret = c.pct_change()

    feats = pd.DataFrame(index=df.index)
    # tendências / momento
    sma_f = c.rolling(cfg.sma_fast, min_periods=1).mean()
    sma_s = c.rolling(cfg.sma_slow, min_periods=1).mean()
    feats["sma_diff"]  = sma_f - sma_s
    feats["sma_ratio"] = (sma_f / (sma_s + 1e-12)) - 1.0
    feats["rsi"] = _rsi(c, cfg.rsi_len)

    # volatilidade e retornos defasados
    feats["vol"] = ret.rolling(cfg.vol_win).std()
    for k in range(1, cfg.n_lags + 1):
        feats[f"ret_lag{k}"] = ret.shift(k)

    # alvo: +1 se próximo retorno > 0
    y = (ret.shift(-1) > 0).astype(int)

    X = feats.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]
    return X, y, list(X.columns)

# ------------------- Treino / Avaliação -------------------

@dataclass
class ModelConfig:
    kind: str = "RandomForest"  # "Logistic" | "RandomForest"
    # RF
    n_estimators: int = 300
    max_depth: int | None = None
    # Logistic
    C: float = 1.0

def train_and_eval(
    df: pd.DataFrame,
    fcfg: FeatureConfig,
    mcfg: ModelConfig,
    prob_threshold: float = 0.55
) -> Dict:
    """
    Treina em janela 80%/20% temporal e devolve métricas + prob de alta do próximo pregão.
    """
    X, y, feats = build_features(df, fcfg)
    if len(X) < 100:
        return dict(error="Dados insuficientes para treinar (<100 amostras).")

    split = int(len(X) * 0.8)
    Xtr, ytr = X.iloc[:split], y.iloc[:split]
    Xte, yte = X.iloc[split:], y.iloc[split:]

    if mcfg.kind == "Logistic":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=mcfg.C, solver="lbfgs"))
        ])
        pipe.fit(Xtr, ytr)
        proba_te = pipe.predict_proba(Xte)[:, 1]
        proba_last = float(pipe.predict_proba(X.iloc[[-1]])[0, 1])
        importances = pd.Series(pipe.named_steps["clf"].coef_[0], index=feats)
    else:  # RandomForest
        clf = RandomForestClassifier(
            n_estimators=mcfg.n_estimators,
            max_depth=mcfg.max_depth,
            n_jobs=-1,
            random_state=42,
            class_weight=None
        )
        clf.fit(Xtr, ytr)
        proba_te = clf.predict_proba(Xte)[:, 1]
        proba_last = float(clf.predict_proba(X.iloc[[-1]])[0, 1])
        importances = pd.Series(clf.feature_importances_, index=feats)

    # métricas
    yhat = (proba_te >= prob_threshold).astype(int)
    acc = accuracy_score(yte, yhat)
    f1 = f1_score(yte, yhat, zero_division=0)
    try:
        auc = roc_auc_score(yte, proba_te)
    except Exception:
        auc = np.nan

    sig = int(proba_last >= prob_threshold)

    return dict(
        acc=float(acc), f1=float(f1), auc=float(auc),
        proba_last=proba_last, signal=sig,
        importances=importances.sort_values(ascending=False),
        n_train=len(Xtr), n_test=len(Xte)
    )
