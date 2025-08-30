# Quant Analyzer — Skeleton

Multi-page Streamlit + Yahoo Finance quantitative analyzer.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
```

## Structure
```
quant_analyzer/
  app.py
  core/
    data.py         # Yahoo Finance helpers
    ui.py           # UI helpers and session defaults
    indicators.py   # simple SMA/RSI (placeholders)
  pages/
    1_🏠_Home.py
    2_📈_Price_Charts.py
    3_🔎_Screener.py
    4_📊_Backtest.py
    5_🧠_ML_Models.py
    6_⚖️_Risk.py
    7_🧺_Portfolio.py
    8_🎲_MonteCarlo.py
    9_⚙️_Settings.py
  data/
    watchlist.json
  requirements.txt
```

> Cada página está como *placeholder*. Vamos implementar módulos um a um após sua aprovação.
