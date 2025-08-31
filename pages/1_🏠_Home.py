# pages/0_🏠_Home.py
from __future__ import annotations
import streamlit as st
from datetime import datetime

from core.ui import app_header, next_steps_card

# --------------------------------------------------------------------
# Config & Header
# --------------------------------------------------------------------
st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")  # docs: st.set_page_config
app_header("🏠 Home", "Visão geral do projeto & Manual Operacional")

st.success("Este é o *esqueleto* do Quant Analyzer. Nada aqui é recomendação; use para estudo e prototipagem.")

# --------------------------------------------------------------------
# Manual (Markdown central + componentes visuais nas abas)
# --------------------------------------------------------------------
MANUAL_MD = f"""
# Manual Operacional — Quant Analyzer

Última atualização: **{datetime.now().strftime("%Y-%m-%d %H:%M")}**

## 1) Visão geral
- App multipáginas em Streamlit; cada página define `st.set_page_config` no topo.
- Dados vindos do Yahoo Finance via camada `core.data` (tratando `Close/Adj Close`, MultiIndex).
- Gráficos com Plotly; tema claro/escuro controlado nas páginas.
- Cache: `st.cache_data` para downloads e transformações; limpe quando mudar versão do data layer.

## 2) Estrutura de páginas
- **Screener**: seleção de ativos; salva no `st.session_state`. Use chaves: `screener_selected` **ou** `screener_selection`.
- **Portfolio (Monte Carlo)**: cenários MVN/Bootstrap, fan chart, probabilidades.
- **Portfolio (Markowitz)**: fronteira eficiente, MVP, tangência, comparação de carteiras.
- **Portfolio Backtest**: buy&hold com rebalance, custos e benchmark.

## 3) Fluxo rápido
1. Marque os ativos no **Screener**.
2. Nas páginas, ative “Usar seleção do Screener”.
3. Ajuste período/intervalo e **pesos** (Equal-Weight, Máx. Sharpe, Manual).
4. Leia KPIs (CAGR, Vol, Sharpe, MaxDD) e exporte CSVs (pesos/equity/percentis).

## 4) Convenções de dados
- Períodos: `6mo`, `1y`, `2y`, `5y` | Intervalos: `1d`, `1wk`.
- Preferir `Close`; se ausente, cair para `Adj Close`.
- Alinhamento por **interseção** de datas (inner join).
- Retornos: `pct_change().dropna()`; anualização: média × 252, vol × √252.

## 5) Cache & Estado
- `@st.cache_data(ttl=600)` em funções de I/O e pré-processo.
- `st.session_state` para seleção do screener e preferências por sessão.

## 6) UI/UX
- Use **abas** para seções longas, **expanders** para detalhes.
- Ofereça **download** (CSV/MD) onde fizer sentido.
- Evite bloquear a UI: envolva operações pesadas em `st.spinner(...)`.

## 7) Erros comuns & correções
- *“If using all scalar values, you must pass an index”*: monte DataFrames com `pd.concat([...], axis=1)` a partir de **Series** nomeadas.
- *Colunas MultiIndex do provedor*: extraia `Close`/`Adj Close` para **Series 1-D** antes de `pct_change()`.
- *Dados insuficientes*: verifique `period/interval` e ativos; mostre aviso e prossiga com os válidos.

"""

# Abas principais para navegação visual (doc oficial: st.tabs)
tab_intro, tab_fluxo, tab_paginas, tab_dados, tab_uiux, tab_erros = st.tabs(
    ["📘 Introdução", "🧭 Fluxo operacional", "🗂️ Páginas", "🗃️ Dados & Cache", "🎛️ UI/UX", "🧯 Erros comuns"]
)

with tab_intro:
    st.markdown("### Introdução")
    st.markdown(MANUAL_MD.split("## 2) Estrutura de páginas")[0])  # só a seção 1 aqui
    next_steps_card([
        "Definir lista de ativos padrão (BR/US/Cripto).",
        "Padronizar funções de dados (Yahoo Finance).",
        "Estabelecer tema visual e componentes de UI.",
    ])
    st.download_button(
        "⬇️ Baixar Manual (Markdown)",
        data=MANUAL_MD.encode("utf-8"),
        file_name="manual_operacional_quant_analyzer.md",
        mime="text/markdown",
        use_container_width=True,  # docs: st.download_button
    )

with tab_fluxo:
    with st.expander("Passo a passo — do Screener ao resultado", expanded=True):  # docs: st.expander
        st.markdown("""
1. **Screener** → selecione tickers.
2. Nas páginas de análise, ative **Usar seleção do Screener**.
3. Escolha **pesos** (Equal-Weight / Máx. Sharpe / Manual).
4. Ajuste **período/intervalo**, rebalance, custos e aportes.
5. Analise **fan chart**, **distribuição final** e **KPIs**; exporte CSVs.
""")
    st.markdown("#### Boas práticas")
    st.markdown("""
- Prefira períodos mais longos para estabilidade de estimativas.
- Documente hipóteses (rf, w_max, rebalance) no relatório/CSV exportado.
""")

with tab_paginas:
    st.markdown("### Estrutura de páginas")
    st.markdown("""
- **Screener**: salva seleção no `session_state` (`screener_selected`/`screener_selection`).
- **Monte Carlo**: cenários (MVN/Bootstrap), probabilidades e export.
- **Markowitz**: fronteira eficiente, MVP, tangência, comparação com Monte Carlo.
- **Backtest**: buy&hold com rebalance, custos e benchmark (SPY, BOVA11.SA, etc.).
""")

with tab_dados:
    st.markdown("### Dados & Cache")
    st.markdown("""
- Extraia `Close` (ou `Adj Close`) e converta para **Series 1-D** (prove dores podem retornar **MultiIndex**).
- Monte DataFrames com `pd.concat(..., axis=1, join="inner")` e `dropna()`.
- Use `@st.cache_data(ttl=600)` em downloads/transformações; limpe ao mudar o data layer.
""")

with tab_uiux:
    st.markdown("### UI/UX")
    st.markdown("""
- Navegue conteúdo longo com **abas**; detalhe técnico em **expanders**.
- Ofereça **st.download_button** para CSV/MD.
- Ajuste tema claro/escuro nos gráficos Plotly via `template`.
""")

with tab_erros:
    st.markdown("### Erros comuns")
    st.markdown("""
- **DataFrame com escalares** → use `pd.concat` de **Series** nomeadas.
- **MultiIndex inesperado** → selecione nível `'Close'`/`'Adj Close'` e converta para Series.
- **Dados insuficientes** → perí odos curtos, ativos sem histórico ou fuso/feriados; trate e informe ao usuário.
""")

# Rodapé compacto do manual completo
with st.expander("📄 Ver o Manual completo (Markdown)", expanded=False):
    st.markdown(MANUAL_MD)


# ---- Home com Manual do Neural Forecast (cole na Home) ----
import streamlit as st

try:
    from core.ui import app_header
    _has_app_header = True
except Exception:
    _has_app_header = False

st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")
(app_header("🏠 Home", "Visão geral e manuais") if _has_app_header else st.title("🏠 Home"))

tabs = st.tabs(["📌 Introdução", "🧠 Manual — Neural Forecast"])

with tabs[0]:
    st.markdown("""
**Bem-vindo!** Aqui você encontra os manuais das páginas do app.
Use a aba **Neural Forecast** para um guia detalhado de previsão com RNAs.
""")

with tabs[1]:
    st.markdown("# 🧠 Manual — Neural Forecast")
    st.caption("Previsão multi-modelo (MLP, LSTM, GRU, CNN-1D, TCN, Transformer) + avaliação no teste + projeção H passos com incerteza via MC Dropout.")

    st.markdown("## 1) O que esta página faz")
    st.markdown("""
- Baixa **OHLCV** do ticker (Yahoo).  
- Constrói **features** (retornos, médias, RSI, volatilidade, etc.).  
- Separa dados em **Treino → Validação → Teste** (ordem temporal, sem shuffle).  
- Treina os modelos selecionados.  
- Compara no **Teste** (MAE, RMSE, MAPE, direcional).  
- Projeta o futuro por **H passos** com **MC Dropout**, gerando **fan chart** (P5…P95) e **P(Δ>0)** por horizonte.
""")

    with st.expander("### 2) Entendendo cada controle", expanded=True):
        st.markdown("""
- **Ticker**: qualquer ativo suportado pelo Yahoo (ex.: `ETH-USD`, `AAPL`, `PETR4.SA`).  
- **Período** / **Intervalo**: tamanho da amostra e resolução (ex.: `2y` + `1d`).  
- **Alvo**  
  - **Log-return** (recomendado): aprende variação; melhor para probabilidade de alta.  
  - **Close (nível)**: aprende o nível de preço.
- **Janela (lookback)**: barras que entram como entrada da rede (janelas deslizantes).  
  - Diário: **30–120**; Intraday: **60–240**.
- **Horizonte de previsão (H)**: quantos passos no futuro (ex.: 10–20).  
- **Proporção de Teste / Validação**: splits temporais (padrões: teste 0.20, val 0.10).  
- **Modelos**: MLP, LSTM, GRU, CNN-1D, TCN (residual dilatado), Transformer (mínimo).  
- **Modo de treino**:  
  - **Rápido** ≈ 25 épocas (bom para explorar).  
  - **Completo** ≈ 80 épocas (melhor desempenho; mais lento).
- **Amostras MC Dropout**: 100–300 costuma ser suficiente.  
- **Seed**: fixa resultados (ideal para comparar rodadas).
""")

    with st.expander("### 3) Pipeline de dados e treino (o que rola nos bastidores)"):
        st.markdown("""
1. **Pré-processamento**: `auto_adjust` de preços, remoção de `NaN`, tz aware → naive.  
2. **Features**: `ret1`, `logret1`, `sma/ema`, `vol20`, `rsi14` (você pode estender).  
3. **Split temporal**: Treino (70%–85%), Val (10% do treino), Teste (15%–30%).  
4. **Normalização**: fit **só** no treino; aplica no val/teste/futuro.  
5. **Janela deslizante**: cria tensores `(amostras, lookback, n_features)` para 1-passo.  
6. **Treino**: EarlyStopping + ReduceLROnPlateau; perda MSE/Huber.  
7. **Backtest 1-passo** no teste (sem look-ahead) → métricas.  
8. **Projeção futura**: recursiva, com **dropout ativo** (MC) para quantis e `P(Δ>0)`.
""")

    with st.expander("### 4) Como interpretar os gráficos e tabelas", expanded=True):
        st.markdown("""
- **Faixas TREINO / VAL / TESTE**: regiões coloridas no histórico.  
- **Tabela de métricas** (Teste):  
  - **RMSE**/**MAE**: erro médio (quanto menor, melhor).  
  - **MAPE**: erro percentual (cuidado com zeros).  
  - **Direcional**: % de acertos de sinal (se alvo for retorno).  
- **Fan chart (futuro)**:  
  - **P50** = mediana (cenário base).  
  - **P25–P75**: região central.  
  - **P5–P95**: extremos plausíveis.  
- **P(Δ>0) por horizonte**: probabilidade de alta a cada `h ∈ [1..H]`.  
  Use como **evidência**, não garantia. Combine com tape/fluxo e gestão de risco.
""")

    with st.expander("### 5) Receitas rápidas (valores sugeridos)"):
        st.markdown("""
**Diário (ações/ETF/cripto)**  
- *Exploração rápida:* `Log-return`, lookback **60**, H **10–20**, Teste **0.20**, Val **0.10**, **todos os modelos**, **Rápido**, MC=**100**.  
- *Rodada para decisão:* **Completo**, MC=**200–300**. Compare melhores (RMSE/MAE) e probabilidade.

**Intraday (1h/15m)**  
- Lookback **120–240**, H **8–16**, MC **150–300**. Dados intraday são ruidosos → foque em horizontes curtos.

**Mercados com forte regime**  
- Aumente período (ex.: `5y`) e valorize modelos **LSTM/TCN/Transformer**.
""")

    with st.expander("### 6) Boas práticas de risco"):
        st.markdown("""
- Use **P25** e **P5** do seu horizonte para calibrar stop e tamanho de posição; **P50** para alvo.  
- Se o preço real começar a rodar **abaixo de P25** de forma consistente, trate como **alerta** (mudança de regime / modelo fora).  
- Re-treine quando chegar **novo bloco de dados** ou quando o mercado **escapar do leque**.
""")

    with st.expander("### 7) Limitações importantes"):
        st.markdown("""
- Métricas do **teste** refletem **um período histórico**; fora da amostra tudo pode mudar.  
- **MC Dropout** modela parte da incerteza, mas **não todos** os riscos (eventos/gaps/liquidez).  
- Alvo `Close (nível)` torna a leitura de `P(Δ>0)` menos direta do que `Log-return`.
""")

    with st.expander("### 8) Solução de problemas (erros comuns)"):
        st.markdown("""
- **`tabulate` ausente** ao salvar Markdown → adicione `tabulate>=0.9` no `requirements.txt` ou ative fallback CSV.  
- **`UnhashableParamError`** (cache) → não cacheie funções que recebem `numpy.ndarray` (remova `@st.cache_*` do `fit`).  
- **Erro em TCN (Add shapes)** → alinhe canais com `Conv1D(64,1)` no atalho antes do `Add`.  
- **TensorFlow ausente** → instale variante correta no `requirements.txt` (CPU: `tensorflow>=2.16,<2.18`; Apple: `tensorflow-macos` + `tensorflow-metal`).
""")

    st.info("**Dica:** comece com `Log-return`, lookback 60, H 10–20, Teste 20%, Val 10%, todos os modelos em 'Rápido'. Se gostar do resultado, rode 'Completo' para refinar.")

