# TSLA AI Indicator — Project Structure
```
tsla_ai_indicator/
│
├── .env                        ← Your SECRET API keys (never share this)
├── .gitignore                  ← Protects .env and auto-gen files from Git
├── requirements.txt            ← All Python packages needed
│
├── data/                       ← Downloaded CSV datasets go here
│   └── tsla_5m_historical.csv  ← Created after running step1
│
├── models/                     ← Trained ML model files (.pkl) go here
│
├── logs/                       ← Runtime prediction logs go here
│
└── src/                        ← All Python scripts (one per phase)
    ├── step1_download_data.py  ← Phase 0: Download & verify TSLA data  ← START HERE
    ├── step2_features.py       ← Phase 1: Feature engineering (RSI, MACD, etc.)
    ├── step3_train_model.py    ← Phase 2: XGBoost model training & validation
    ├── step4_backtest.py       ← Phase 2: Walk-forward backtesting
    └── step5_app.py            ← Phase 3-4: FastAPI live dashboard
```

## Quick Start (Do this once)
```bash
cd tsla_ai_indicator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# → Fill in your Alpaca keys in .env
python src/step1_download_data.py
```
