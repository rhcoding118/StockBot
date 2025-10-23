#!/usr/bin/env python3
"""
train_model.py — Train the PriceLSTM used by bot.py (Multi-Symbol Compatible)

Example:
  python train_model.py \
    --symbols AAPL MSFT NVDA \
    --start 2016-01-01 --end 2024-12-31 \
    --epochs 8 --horizon 5 --pos-thr 0.01 --neg-thr -0.01 \
    --out-path ./models/model.pt

This script:
  - Downloads OHLCV with yfinance
  - Builds features identical to bot.py (shape: T=32, F=16)
  - Labels next-k-day return: SELL/HOLD/BUY
  - Trains a small LSTM classifier (output: 3 classes)
  - Saves model state_dict to ./models/model.pt

Dependencies:
  pip install torch yfinance numpy pandas scikit-learn
"""

import os
import math
import argparse
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------
# Model (must match bot.py)
# ----------------------------
class PriceLSTM(nn.Module):
    def __init__(self, input_dim=16, hidden=32, layers=1, output_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):  # x: (B, T, F)
        y, _ = self.lstm(x)
        logits = self.head(y[:, -1, :])
        return logits

# ----------------------------
# Feature builder (matches bot.py)
# ----------------------------
def build_features_from_prices(price_window: np.ndarray) -> np.ndarray:
    """
    Input: price_window shape (32,)
    Output: features shape (32, 16) — must match bot.py
    """
    window = price_window.astype(np.float32)
    # Features: normalized price, z-score, simple returns
    rets = np.diff(window) / np.maximum(window[:-1], 1e-6)
    rets = np.concatenate([rets, [0.0]])  # keep length 32
    z = (window - window.mean()) / (window.std() + 1e-6)
    norm_p = window / np.max(window)

    feats = np.stack([norm_p, z, rets], axis=1)  # (32, 3)
    # Tile to 16 features (simple expansion to fit example model)
    if feats.shape[1] < 16:
        reps = int(np.ceil(16 / feats.shape[1]))
        feats = np.tile(feats, (1, reps))[:, :16]
    return feats

# ----------------------------
# Labeling
# ----------------------------
def label_next_return(closes: np.ndarray, idx: int, horizon: int, pos_thr: float, neg_thr: float) -> int:
    """
    Look ahead 'horizon' steps from idx to compute future return.
    Return label: 0=SELL, 1=HOLD, 2=BUY
    """
    if idx + horizon >= len(closes):
        return -1  # invalid (no label)
    ret = (closes[idx + horizon] - closes[idx]) / max(closes[idx], 1e-6)
    if ret >= pos_thr:
        return 2  # BUY
    elif ret <= neg_thr:
        return 0  # SELL
    else:
        return 1  # HOLD

# ----------------------------
# Dataset
# ----------------------------
class WindowDataset(Dataset):
    def __init__(self, frames: List[pd.DataFrame], window: int, horizon: int, pos_thr: float, neg_thr: float):
        self.X, self.y = [], []
        for df in frames:
            closes = df["Close"].values.astype(np.float32)
            for i in range(window, len(df) - horizon):
                price_window = closes[i - window:i]
                label = label_next_return(closes, i - 1, horizon, pos_thr, neg_thr)
                if label == -1:
                    continue
                feats = build_features_from_prices(price_window)
                self.X.append(feats)
                self.y.append(label)
        self.X = np.stack(self.X, axis=0).astype(np.float32)  # (N, 32, 16)
        self.y = np.array(self.y, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------------------
# Data utilities
# ----------------------------
def download_symbol_frame(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    df = df.dropna()
    if not {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
        raise ValueError(f"Downloaded data missing columns for {symbol}")
    df["Symbol"] = symbol
    return df

def train_val_split(n_samples: int, val_frac: float = 0.2, seed: int = 42):
    set_seed(seed)
    idxs = np.arange(n_samples)
    np.random.shuffle(idxs)
    n_val = int(np.floor(n_samples * val_frac))
    val_idx = idxs[:n_val]
    train_idx = idxs[n_val:]
    return train_idx, val_idx

# ----------------------------
# Training loop
# ----------------------------
@dataclass
class TrainConfig:
    window: int = 32
    horizon: int = 5
    pos_thr: float = 0.01
    neg_thr: float = -0.01
    batch_size: int = 256
    epochs: int = 8
    lr: float = 3e-4
    hidden: int = 32
    layers: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_path: str = "./models/model.pt"
    val_frac: float = 0.2
    seed: int = 42

def train_model(cfg: TrainConfig, symbols: List[str], start: str, end: str):
    os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)
    set_seed(cfg.seed)

    # 1) Load data
    frames = [download_symbol_frame(s, start, end) for s in symbols]
    dataset = WindowDataset(frames, cfg.window, cfg.horizon, cfg.pos_thr, cfg.neg_thr)
    if len(dataset) == 0:
        raise RuntimeError("No training samples built. Try different dates/symbols/thresholds.")
    print(f"Samples: {len(dataset)}  (from {len(symbols)} symbols)")

    # 2) Split
    train_idx, val_idx = train_val_split(len(dataset), cfg.val_frac, cfg.seed)
    train_X = torch.tensor(dataset.X[train_idx], dtype=torch.float32)
    train_y = torch.tensor(dataset.y[train_idx], dtype=torch.long)
    val_X   = torch.tensor(dataset.X[val_idx], dtype=torch.float32)
    val_y   = torch.tensor(dataset.y[val_idx], dtype=torch.long)

    train_loader = DataLoader(list(zip(train_X, train_y)),
                              batch_size=min(len(train_X), cfg.batch_size),
                              shuffle=True)
    val_loader   = DataLoader(list(zip(val_X, val_y)),
                              batch_size=min(len(val_X), cfg.batch_size),
                              shuffle=False)

    # 3) Model / Optimizer / Loss
    model = PriceLSTM(input_dim=16, hidden=cfg.hidden, layers=cfg.layers, output_dim=3).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 4) Train
    best_val_acc, best_state = 0.0, None
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, total, correct = 0.0, 0, 0
        for Xb, yb in train_loader:
            Xb = Xb.to(cfg.device)
            yb = yb.to(cfg.device)
            opt.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss += loss.item() * Xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += Xb.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # Validate
        model.eval()
        v_total, v_correct = 0, 0
        all_pred, all_true = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(cfg.device)
                yb = yb.to(cfg.device)
                logits = model(Xb)
                pred = logits.argmax(dim=1)
                v_correct += (pred == yb).sum().item()
                v_total += Xb.size(0)
                all_pred.extend(pred.cpu().numpy().tolist())
                all_true.extend(yb.cpu().numpy().tolist())

        val_acc = v_correct / max(v_total, 1)
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # 5) Save best
    if best_state is None:
        best_state = model.state_dict()
    torch.save(best_state, cfg.out_path)
    print(f"Saved best model to {cfg.out_path} (val_acc={best_val_acc:.3f})")

    # 6) Simple report on validation split
    with torch.no_grad():
        model.load_state_dict(best_state)
        logits_chunks = []
        for i in range(0, len(val_X), 2048):
            logits_chunks.append(model(val_X[i:i+2048].to(cfg.device)).cpu())
        logits = torch.cat(logits_chunks, dim=0)
        preds = logits.argmax(dim=1).numpy()
        truth = val_y.numpy()
        print("\nValidation report (0=SELL, 1=HOLD, 2=BUY):")
        print(classification_report(truth, preds, digits=3, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(truth, preds))

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train PriceLSTM for stock action classification")
    ap.add_argument("--symbols", nargs="+", required=True, help="e.g., AAPL MSFT NVDA")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=5, help="Lookahead days for label")
    ap.add_argument("--pos-thr", type=float, default=0.01, help="BUY if future return >= pos_thr")
    ap.add_argument("--neg-thr", type=float, default=-0.01, help="SELL if future return <= neg_thr")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--out-path", default="./models/model.pt")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        window=args.window,
        horizon=args.horizon,
        pos_thr=args.pos_thr,
        neg_thr=args.neg_thr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        layers=args.layers,
        out_path=args.out_path,
        seed=args.seed,
    )
    train_model(cfg, symbols=args.symbols, start=args.start, end=args.end)
