# DL-Trader (Deep Learning • Paper & Live via E*TRADE)

A minimal, production-ready scaffold for a **deep-learning trading bot** that runs in:
- **Paper mode** (local JSON ledger; no real orders)
- **Live mode** via **E*TRADE** API (OAuth 1.0a; sandbox or production)

It supports **multiple symbols**, basic risk controls, and a **one-flag switch** between paper and real trading.

> ⚠️ Educational use only. Not investment advice. Always validate behavior in **paper** and **sandbox** before using real money.

---

## Contents

- [`bot.py`](./bot.py): trading loop, risk management, brokers
- [`train_model.py`](./train_model.py): trains the `PriceLSTM` used by the bot
- `models/model.pt`: saved PyTorch weights (produced by `train_model.py`)
- `paper_ledger.json`: created automatically in paper mode

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch numpy pandas scikit-learn requests yfinance
