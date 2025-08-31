# pages/11_🧠_Neural_Forecast.py
# =============================================================================
# Neural Forecast: MLP, LSTM, GRU, CNN1D, TCN (simples) e Transformer
# - Marca treino/val/teste no gráfico
# - Tabela de métricas no período de teste
# - Projeção futura multi-horizonte com incerteza (MC Dropout)
# - Probabilidade P(Δ>0) por horizonte
# =============================================================================
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# --------------- TF opcional (fallback seguro) ---------------
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_OK = True
except Exception:
    TF_OK = False

# --------------- Dados ---------------
try:
    import yfinance as yf  # opcional, mas recomendado
except Exception:
    yf = None

st.set_page_config(page_title="Neural Forecast", page_icon="🧠", layout="wide")
st.title("🧠 Neural Forecast")
st.caption("MLP, LSTM, GRU, CNN-1D, TCN simples e Transformer — treino/val/teste + projeção futura com incerteza (MC Dropout)")

# =============================================================================
# Sidebar — controles
# =============================================================================
with st.sidebar:
    st.markdown("### Configurações")
    ticker = st.text_input("Ticker", value="AAPL")
    period  = st.selectbox("Período", ["6mo","1y","2y","5y","max"], index=2)
    interval = st.selectbox("Intervalo", ["1d","1h","1wk"], index=0)

    target_kind = st.selectbox("Alvo", ["Close (nível)", "Log-return"], index=1)
    lookback = st.slider("Janela (lookback)", 20, 200, 60, step=5)
    horizon  = st.slider("Horizonte de previsão (passos)", 1, 60, 20, step=1)

    test_frac = st.slider("Proporção de Teste", 0.05, 0.35, 0.2, step=0.05)
    val_frac  = st.slider("Proporção de Validação (dentro do treino)", 0.05, 0.3, 0.1, step=0.05)

    st.markdown("#### Modelos")
    m_mlp = st.checkbox("MLP (Dense)", True)
    m_lstm = st.checkbox("LSTM", True)
    m_gru = st.checkbox("GRU", True)
    m_cnn = st.checkbox("CNN-1D", True)
    m_tcn = st.checkbox("TCN (simples)", True)
    m_trf = st.checkbox("Transformer (mínimo)", True)

    mode = st.selectbox("Modo de treino", ["Rápido", "Completo"], index=0)
    epochs = 25 if mode == "Rápido" else 80
    mc_samples = st.slider("Amostras MC Dropout", 20, 300, 100, step=10)

    seed = st.number_input("Seed", value=42, step=1)
    np.random.seed(seed)
    if TF_OK:
        tf.random.set_seed(seed)

# =============================================================================
# Utilidades
# =============================================================================
def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / (ma_dn + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

@st.cache_data(ttl=600, show_spinner=False)
def dl_prices(sym: str, period: str, interval: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        df = yf.download(sym, period=period, interval=interval, auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.title)
    df = df.dropna()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["close"] = df["Close"].astype(float)

    # target opções
    if target_kind.startswith("Log"):
        out["ret"] = np.log(out["close"]).diff()
        y_col = "ret"
    else:
        y_col = "close"

    # features básicas
    out["ret1"] = out["close"].pct_change()
    out["logret1"] = np.log(out["close"]).diff()
    out["sma10"] = out["close"].rolling(10, min_periods=5).mean()
    out["sma20"] = out["close"].rolling(20, min_periods=10).mean()
    out["ema10"] = out["close"].ewm(span=10, adjust=False).mean()
    out["vol20"] = out["ret1"].rolling(20).std()
    out["rsi14"] = rsi(out["close"], 14)
    out = out.dropna()
    return out, y_col

def time_split(df: pd.DataFrame, test_frac: float, val_frac: float):
    n = len(df)
    n_test = max(1, int(n * test_frac))
    n_trainval = n - n_test
    n_val = max(1, int(n_trainval * val_frac))
    n_train = n_trainval - n_val
    idx_train = df.index[:n_train]
    idx_val   = df.index[n_train:n_train+n_val]
    idx_test  = df.index[n_train+n_val:]
    return idx_train, idx_val, idx_test

def make_windows(X: np.ndarray, y: np.ndarray, L: int, step: int = 1):
    """
    1-step ahead janelas com lookback L (para treinar multi-step recursivo).
    Retorna X3D (samples, L, n_features) e y1D (samples,)
    """
    xs, ys = [], []
    for i in range(L, len(X), step):
        xs.append(X[i-L:i])
        ys.append(y[i])
    if not xs:
        return np.empty((0, L, X.shape[1])), np.empty((0,))
    return np.stack(xs), np.array(ys)

def scale_fit(train: np.ndarray):
    mu = train.mean(axis=0)
    sd = train.std(axis=0) + 1e-12
    return mu, sd

def scale_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (x - mu) / sd

def inv_transform_levels(last_level: float, y_ret_path: np.ndarray) -> np.ndarray:
    """Caso alvo seja log-retorno: reconstrói níveis a partir de último nível + cumulativos."""
    levels = np.empty_like(y_ret_path, dtype=float)
    acc = last_level
    for t in range(y_ret_path.shape[0]):
        acc = acc * np.exp(y_ret_path[t])
        levels[t] = acc
    return levels

# =============================================================================
# Carregar & preparar dados
# =============================================================================
with st.spinner("📥 Baixando dados..."):
    raw = dl_prices(ticker, period, interval)

if raw.empty:
    st.error("Sem dados. Verifique ticker/período/intervalo.")
    st.stop()

feats, y_col = make_features(raw)
idx_train, idx_val, idx_test = time_split(feats, test_frac, val_frac)

# matrizes
X_all = feats.drop(columns=[y_col]).to_numpy(dtype=np.float32)
y_all = feats[[y_col]].to_numpy(dtype=np.float32).squeeze(-1)

# normalização (fit no treino)
mu_x, sd_x = scale_fit(X_all[feats.index.get_indexer(idx_train)])
X_scaled = scale_apply(X_all, mu_x, sd_x)

# janelas para treino/val/teste (1-step ahead)
X3, y1 = make_windows(X_scaled, y_all, lookback, step=1)
# alinha índices das amostras com a série original (a partir de lookback)
idx_all = feats.index[lookback:]

# fatias
def slice_by_index(idx_slice):
    mask = idx_all.isin(idx_slice)
    return X3[mask], y1[mask], idx_all[mask]

Xtr, ytr, idx_tr = slice_by_index(idx_train)
Xva, yva, idx_va = slice_by_index(idx_val)
Xte, yte, idx_te = slice_by_index(idx_test)

n_features = Xtr.shape[-1]
if min(len(Xtr), len(Xva), len(Xte)) < 10:
    st.warning("Muito poucos pontos para treinar/testar com esses parâmetros. Tente reduzir lookback ou aumentar período.")
    
# =============================================================================
# Modelos Keras (opcionais; fallback se TF ausente)
# =============================================================================
MODELS = {}

def m_mlp_dense():
    inp = layers.Input(shape=(lookback, n_features))
    x = layers.Flatten()(inp)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out, name="MLP")
    return model

def m_lstm_seq():
    inp = layers.Input(shape=(lookback, n_features))
    x = layers.LSTM(64, return_sequences=True, dropout=0.2)(inp)
    x = layers.LSTM(32, dropout=0.2)(x)
    out = layers.Dense(1)(x)
    return keras.Model(inp, out, name="LSTM")

def m_gru_seq():
    inp = layers.Input(shape=(lookback, n_features))
    x = layers.GRU(64, return_sequences=True, dropout=0.2)(inp)
    x = layers.GRU(32, dropout=0.2)(x)
    out = layers.Dense(1)(x)
    return keras.Model(inp, out, name="GRU")

def m_cnn1d():
    inp = layers.Input(shape=(lookback, n_features))
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(64, 5, dilation_rate=2, padding="causal", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1)(x)
    return keras.Model(inp, out, name="CNN1D")

def m_tcn_simple():
    # TCN minimalista (residual conv dilatada)
    inp = layers.Input(shape=(lookback, n_features))
    x = inp
    for d in [1, 2, 4]:
        h = layers.Conv1D(64, 3, dilation_rate=d, padding="causal", activation="relu")(x)
        h = layers.Dropout(0.2)(h)
        h = layers.Conv1D(64, 3, dilation_rate=d, padding="causal", activation="relu")(h)
        x = layers.Add()([x, h])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1)(x)
    return keras.Model(inp, out, name="TCN")

def m_transformer_min():
    # Transformer encoder mínimo
    inp = layers.Input(shape=(lookback, n_features))
    x = layers.LayerNormalization()(inp)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.2)(x, x, training=True)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(64, 1, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1)(x)
    return keras.Model(inp, out, name="Transformer")

if TF_OK:
    if m_mlp: MODELS["MLP"] = m_mlp_dense
    if m_lstm: MODELS["LSTM"] = m_lstm_seq
    if m_gru: MODELS["GRU"] = m_gru_seq
    if m_cnn: MODELS["CNN1D"] = m_cnn1d
    if m_tcn: MODELS["TCN"] = m_tcn_simple
    if m_trf: MODELS["Transformer"] = m_transformer_min
else:
    st.warning("TensorFlow não disponível — usando baseline (naïve). Instale `tensorflow` para habilitar as RNAs.")

# =============================================================================
# Treino & avaliação
# =============================================================================
@st.cache_resource(show_spinner=False)
def compile_fit(model_fn, Xtr, ytr, Xva, yva, epochs: int, seed: int):
    model = model_fn()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    cb = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]
    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=epochs, batch_size=64, verbose=0, callbacks=cb,
    )
    return model

def predict_1step(model, X):
    # usar dropout também em inferência (training=True) p/ MC
    yhat = model(X, training=True).numpy().squeeze()
    return yhat

def one_step_backtest(model, Xte, yte):
    ypred = predict_1step(model, Xte)
    return ypred

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9)))) * 100
    # direcional (apenas se alvo for retorno)
    try:
        diracc = float(np.mean(np.sign(y_true) == np.sign(y_pred))) * 100
    except Exception:
        diracc = np.nan
    return dict(MAE=mae, RMSE=rmse, MAPE=mape, DirAcc=diracc)

# Treino
trained = {}
results = []

if TF_OK and MODELS:
    with st.spinner("🧠 Treinando modelos..."):
        for name, fn in MODELS.items():
            mdl = compile_fit(fn, Xtr, ytr, Xva, yva, epochs=epochs, seed=seed)
            yhat_te = one_step_backtest(mdl, Xte, yte)
            m = metrics(yte, yhat_te)
            trained[name] = dict(model=mdl, yhat_te=yhat_te, metrics=m)
            results.append({"Modelo": name, **m})
else:
    # Baseline: naïve = próximo valor igual ao último observado
    yhat_te = Xte[:, -1, 0] * 0.0 + yte.mean() if target_kind.startswith("Log") else Xte[:, -1, 0] * 0 + feats["close"].iloc[-1]
    m = metrics(yte, yhat_te)
    trained["Naive"] = dict(model=None, yhat_te=yhat_te, metrics=m)
    results.append({"Modelo": "Naive", **m})

# Tabela de métricas do teste
met_df = pd.DataFrame(results).sort_values("RMSE")
st.subheader("Métricas no período de teste")
st.dataframe(met_df, use_container_width=True, height=220)

best_name = met_df.iloc[0]["Modelo"]
best_obj = trained[best_name]
st.caption(f"Melhor modelo no teste: **{best_name}** (RMSE {best_obj['metrics']['RMSE']:.4f}, MAE {best_obj['metrics']['MAE']:.4f})")

# =============================================================================
# Projeção futura com MC Dropout
# =============================================================================
def recursive_forecast_mc(model, last_window: np.ndarray, H: int, mcN: int) -> np.ndarray:
    """
    last_window: (lookback, n_features) já escalonado
    retorna samples x H de previsões do ALVO (retorno ou nível conforme o treino)
    """
    samples = np.zeros((mcN, H), dtype=float)
    for s in range(mcN):
        w = last_window.copy()[None, ...]  # (1, L, F)
        preds = []
        for h in range(H):
            yhat = model(w, training=True).numpy().reshape(-1)[0]  # 1-step com dropout ativo
            preds.append(yhat)
            # avança janela: injeta previsão como proxy em 1 feature (aqui: ret1/logret1)
            # estratégia simples: atualiza apenas a 1ª feature (ret1) com yhat (se alvo for retorno)
            next_step = w[0, 1:, :].copy()
            if y_col != "close":
                # alvo são retornos: colocamos yhat na coluna 0 (ret1/logret1)
                next_feat = w[0, -1, :].copy()
                next_feat[0] = yhat  # assume 1ª feature ~ retorno
                next_step = np.vstack([next_step, next_feat])
            else:
                # alvo é nível: mantemos features (ingênuo). Alternativa: reconstrua features.
                next_step = np.vstack([next_step, w[0, -1, :]])
            w = next_step[None, ...]
        samples[s, :] = preds
    return samples

if TF_OK and best_obj["model"] is not None:
    last_win = X_scaled[-lookback:, :]  # (L, F)
    mc_samples_arr = recursive_forecast_mc(best_obj["model"], last_win, horizon, mc_samples)
    # quantis
    P5  = np.percentile(mc_samples_arr, 5, axis=0)
    P25 = np.percentile(mc_samples_arr, 25, axis=0)
    P50 = np.percentile(mc_samples_arr, 50, axis=0)
    P75 = np.percentile(mc_samples_arr, 75, axis=0)
    P95 = np.percentile(mc_samples_arr, 95, axis=0)

    # probabilidade de Δ>0 por horizonte
    prob_up = (mc_samples_arr > 0).mean(axis=0) if y_col != "close" else ((mc_samples_arr - mc_samples_arr[:, :1]) > 0).mean(axis=0)
else:
    # baseline: ruído branco centrado no último retorno
    mc = np.random.normal(loc=0.0, scale=np.std(yte) if len(yte) else 0.01, size=(mc_samples, horizon))
    P5, P25, P50, P75, P95 = np.percentile(mc, [5,25,50,75,95], axis=0)
    prob_up = (mc > 0).mean(axis=0)

# Reconstrução para plot se alvo for retorno: projetar níveis
last_close = feats["close"].iloc[-1]
if y_col != "close":
    # usar mediana P50 para níveis, e faixas também
    def path_to_levels(path):
        return inv_transform_levels(last_close, path)
    L5, L25, L50, L75, L95 = [path_to_levels(x) for x in [P5, P25, P50, P75, P95]]
else:
    # se modelou níveis diretamente, mostra níveis previstos (simplificação)
    L5, L25, L50, L75, L95 = P5, P25, P50, P75, P95

# =============================================================================
# Gráfico: histórico (treino/val/teste) + previsão futura
# =============================================================================
st.subheader("Séries: treino / validação / teste e projeção")

fig = go.Figure()

# histórico real (Close)
fig.add_trace(go.Scatter(
    x=feats.index, y=feats["close"], mode="lines", name="Preço (Close)", line=dict(width=1.8)
))

# previsões 1-step no teste do melhor modelo
if TF_OK and best_obj["model"] is not None:
    # Precisamos mapear y_pred do teste para nível se alvo for retorno
    ypred_te = best_obj["yhat_te"]
    if y_col != "close":
        # converter para níveis: acumular retornos sobre janelas correspondentes
        # Como é 1-step, aproximamos partindo de close alinhado
        # nível referência = close no início da série de teste (lookback shift)
        # Para um overlay mais simples, vamos plotar apenas o sinal direcional: reconstrução local
        # (representação: normalizamos previsão para escala do y_te real)
        pass  # para simplicidade, não convertemos aqui; mantemos apenas fan chart futuro
else:
    pass

# sombrear regiões
x0 = feats.index.min()
x1 = idx_train.max() if len(idx_train) else feats.index[0]
x2 = idx_val.max() if len(idx_val) else x1
x3 = feats.index.max()

fig.add_vrect(x0=x0, x1=x1, fillcolor="#636efa", opacity=0.08, layer="below", line_width=0, annotation_text="Treino", annotation_position="top left")
if len(idx_val):
    fig.add_vrect(x0=idx_val.min(), x1=idx_val.max(), fillcolor="#EF553B", opacity=0.08, layer="below", line_width=0, annotation_text="Validação", annotation_position="top left")
if len(idx_test):
    fig.add_vrect(x0=idx_test.min(), x1=idx_test.max(), fillcolor="#00cc96", opacity=0.08, layer="below", line_width=0, annotation_text="Teste", annotation_position="top left")

# projeção futura (fan chart)
future_idx = pd.date_range(start=feats.index[-1], periods=horizon+1, freq=("D" if interval=="1d" else ("H" if interval=="1h" else "W")))
future_idx = future_idx[1:]

fig.add_trace(go.Scatter(x=future_idx, y=L95, line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=future_idx, y=L75, fill="tonexty", name="75–95%", mode="lines"))
fig.add_trace(go.Scatter(x=future_idx, y=L50, fill="tonexty", name="50–75%", mode="lines"))
fig.add_trace(go.Scatter(x=future_idx, y=L25, fill="tonexty", name="25–50%", mode="lines"))
fig.add_trace(go.Scatter(x=future_idx, y=L5,  fill="tonexty", name="5–25%",  mode="lines"))

fig.update_layout(
    template="plotly_dark" if st.get_option("theme.base")=="dark" else "plotly_white",
    xaxis_title="Tempo",
    yaxis_title="Preço (nível)",
    margin=dict(l=10,r=10,t=40,b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    title=f"{ticker} — Treino/Val/Teste + Projeção (MC Dropout)",
)
st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# Probabilidade por horizonte
# =============================================================================
st.subheader("Probabilidade por horizonte (P(Δ>0))")
prob_df = pd.DataFrame({
    "h": np.arange(1, horizon+1),
    "P_up": prob_up
})
st.line_chart(prob_df.set_index("h"))

# =============================================================================
# Resumo executivo + export
# =============================================================================
def _fmt_pct(x, nd=1):
    try: return f"{float(x)*100:.{nd}f}%"
    except: return "—"

best = best_name
rmse = best_obj["metrics"]["RMSE"]
mae  = best_obj["metrics"]["MAE"]
st.markdown(f"**Resumo** — Melhor modelo: **{best}** • RMSE **{rmse:.5f}**, MAE **{mae:.5f}**.  \
No cenário futuro, a probabilidade mediana de alta no curto prazo é **{_fmt_pct(prob_up[:max(1,min(5,horizon))].mean())}**.")

# tabela de quantis + probabilidades
quant_tbl = pd.DataFrame({
    "h": np.arange(1, horizon+1),
    "P(Δ>0)": prob_up,
    "P5": L5, "P25": L25, "P50": L50, "P75": L75, "P95": L95,
})
st.dataframe(quant_tbl, use_container_width=True, height=280)

# CSV
st.download_button(
    "⬇️ Baixar Tabela de Projeção (CSV)",
    data=quant_tbl.to_csv(index=False).encode("utf-8"),
    file_name=f"{ticker}_neural_forecast.csv",
    mime="text/csv",
    use_container_width=True,
)

# Markdown
md = []
md.append(f"# Neural Forecast — {ticker}")
md.append(f"- Período: **{period}**, Intervalo: **{interval}**, Lookback: **{lookback}**, Horizonte: **{horizon}**")
md.append(f"- Modelos treinados: **{', '.join(MODELS.keys()) if TF_OK else 'Naive'}**")
md.append(f"- Melhor modelo (teste): **{best}** — RMSE **{rmse:.5f}**, MAE **{mae:.5f}**")
md.append(f"- Probabilidade média de alta (h=1..min(5,H)): **{_fmt_pct(prob_up[:max(1,min(5,horizon))].mean())}**")
md.append("")
md.append("## Quantis & Probabilidades")
md.append(quant_tbl.to_markdown(index=False))
st.download_button(
    "⬇️ Baixar Resumo (Markdown)",
    data="\n".join(md).encode("utf-8"),
    file_name=f"{ticker}_neural_forecast.md",
    mime="text/markdown",
    use_container_width=True,
)
