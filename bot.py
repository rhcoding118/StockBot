#!/usr/bin/env python3
"""
Deep-Learning Trading Bot (Paper | Live E*TRADE) — Multi-Symbol

Examples:
  python bot.py --mode paper --symbols AAPL MSFT --model-path ./models/model.pt
  python bot.py --mode live  --symbols VTI --model-path ./models/model.pt --units 10

Notes:
- Paper mode tracks cash and positions locally (JSON ledger).
- Live mode uses E*TRADE API (OAuth 1.0a). Fill in keys below.
- Strategy expects a normalized feature vector; adjust `build_features` if your
  training changes.

Env vars:
  ETRADE_CONSUMER_KEY, ETRADE_CONSUMER_SECRET
  ETRADE_OAUTH_TOKEN, ETRADE_OAUTH_TOKEN_SECRET
  ETRADE_BASE_URL  (optional; default: https://apisb.etrade.com  | prod: https://api.etrade.com)
"""

import os
import time
import json
import hmac
import base64
import hashlib
import logging
import random
import string
import urllib.parse as up
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import requests

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("dl-trader")

# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    mode: str = "paper"  # "paper" or "live"
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "MSFT"])
    model_path: str = "./models/model.pt"
    poll_seconds: int = 30
    units: int = 5                    # default shares per trade (per symbol)
    max_position_value: float = 10_000.0
    max_position_pct: float = 0.25    # 25% of equity max per symbol
    stop_loss_pct: float = 0.02        # 2% below entry
    take_profit_pct: float = 0.04      # 4% above entry
    paper_ledger_path: str = "./paper_ledger.json"
    # E*TRADE API (OAuth 1.0a)
    etrade_base: str = os.getenv("ETRADE_BASE_URL", "https://apisb.etrade.com")
    consumer_key: str = os.getenv("ETRADE_CONSUMER_KEY", "")
    consumer_secret: str = os.getenv("ETRADE_CONSUMER_SECRET", "")
    oauth_token: str = os.getenv("ETRADE_OAUTH_TOKEN", "")
    oauth_token_secret: str = os.getenv("ETRADE_OAUTH_TOKEN_SECRET", "")

    def validate(self):
        assert self.mode in ("paper", "live")
        if self.mode == "live":
            missing = [k for k, v in {
                "consumer_key": self.consumer_key,
                "consumer_secret": self.consumer_secret,
                "oauth_token": self.oauth_token,
                "oauth_token_secret": self.oauth_token_secret,
            }.items() if not v]
            if missing:
                raise ValueError(f"Missing E*TRADE credentials: {missing}")

# ----------------------------
# Simple LSTM Model (example)
# ----------------------------
class PriceLSTM(nn.Module):
    def __init__(self, input_dim=16, hidden=32, layers=1, output_dim=3):
        """
        output_dim: [0=SELL, 1=HOLD, 2=BUY]
        """
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

class DLStrategy:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model = PriceLSTM()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    @torch.no_grad()
    def predict_action(self, features_window: np.ndarray) -> Tuple[str, float]:
        """
        features_window: shape (T, F), normalized.
        Returns: (action, confidence)
        """
        x = torch.tensor(features_window, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        idx = int(np.argmax(probs))
        action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        return action_map[idx], float(probs[idx])

# ----------------------------
# Data feed (stub)
# ----------------------------
class DataFeed:
    """
    Replace with your preferred data source.
    For paper mode, you can use yfinance or cached OHLCV.
    For live mode, consider E*TRADE market data endpoints (requires entitlements).
    """
    def get_quote(self, symbol: str) -> float:
        # Placeholder: random-walk-ish mock to let the bot run in paper mode.
        price = 100 + np.random.randn() * 0.5
        return round(float(price), 2)

    def get_quotes(self, symbols: List[str]) -> Dict[str, float]:
        # TODO: implement a real data source for realistic behavior
        return {s: self.get_quote(s) for s in symbols}

    def build_features(self, prices: List[float]) -> np.ndarray:
        """
        Create a small feature vector from recent prices.
        Customize to your model’s training scheme.
        """
        window = np.array(prices[-32:], dtype=np.float32)  # T=32
        if window.size < 32:
            pad = np.full(32 - window.size, window[-1] if window.size else 100.0, dtype=np.float32)
            window = np.concatenate([pad, window])
        # Features: normalized returns + rolling stats
        rets = np.diff(window) / np.maximum(window[:-1], 1e-6)
        rets = np.concatenate([rets, [0.0]])  # keep length 32
        z = (window - window.mean()) / (window.std() + 1e-6)
        feats = np.stack([window/np.max(window), z, rets], axis=1)  # (32, 3)
        # Pad/expand to model input dim 16 by tiling
        if feats.shape[1] < 16:
            reps = int(np.ceil(16 / feats.shape[1]))
            feats = np.tile(feats, (1, reps))[:, :16]
        return feats  # (32, 16)

# ----------------------------
# Broker abstraction
# ----------------------------
class Broker:
    def get_equity(self) -> float:
        raise NotImplementedError

    def get_position(self, symbol: str) -> Tuple[int, Optional[float]]:
        """returns (units, avg_entry_price)"""
        raise NotImplementedError

    def place_order(self, symbol: str, side: str, quantity: int, price: Optional[float] = None) -> str:
        raise NotImplementedError

    def close_position(self, symbol: str) -> Optional[str]:
        units, _ = self.get_position(symbol)
        if units == 0:
            return None
        side = "SELL" if units > 0 else "BUY"
        return self.place_order(symbol, side, abs(units))

# ----------------------------
# Paper Broker (local ledger)
# ----------------------------
class PaperBroker(Broker):
    def __init__(self, ledger_path: str, starting_cash: float = 100_000.0):
        self.ledger_path = ledger_path
        if not os.path.exists(ledger_path):
            self.state = {
                "cash": starting_cash,
                "positions": {},  # symbol -> {"units": int, "avg_price": float}
                "orders": []
            }
            self._save()
        else:
            with open(ledger_path, "r") as f:
                self.state = json.load(f)

    def _save(self):
        with open(self.ledger_path, "w") as f:
            json.dump(self.state, f, indent=2)

    def mark_to_market(self, quotes: Dict[str, float]) -> float:
        equity = self.state["cash"]
        for sym, pos in self.state["positions"].items():
            if sym in quotes:
                equity += pos["units"] * quotes[sym]
        return float(equity)

    def get_equity(self) -> float:
        # This returns last cached equity (cash + last known marks).
        # For more accuracy, call mark_to_market with fresh quotes.
        eq = self.state["cash"]
        for sym, pos in self.state["positions"].items():
            # assume last avg_price as proxy if no fresh quote
            eq += pos["units"] * pos["avg_price"]
        return float(eq)

    def get_position(self, symbol: str) -> Tuple[int, Optional[float]]:
        pos = self.state["positions"].get(symbol)
        if not pos:
            return 0, None
        return int(pos["units"]), float(pos["avg_price"])

    def place_order(self, symbol: str, side: str, quantity: int, price: Optional[float] = None) -> str:
        if quantity <= 0:
            raise ValueError("Quantity must be > 0")
        # Market order: price must be provided by caller (latest quote)
        if price is None:
            raise ValueError("PaperBroker needs a fill price (use latest quote)")

        pos = self.state["positions"].get(symbol, {"units": 0, "avg_price": 0.0})
        order_id = f"paper-{int(time.time())}-{random.randint(1000,9999)}"
        cost = quantity * price

        if side.upper() == "BUY":
            if self.state["cash"] < cost:
                raise ValueError("Insufficient cash for paper trade")
            # new weighted average price
            total_units = pos["units"] + quantity
            if total_units == 0:
                avg = 0.0
            else:
                avg = (pos["avg_price"] * pos["units"] + cost) / total_units
            pos["units"] = total_units
            pos["avg_price"] = avg
            self.state["cash"] -= cost

        elif side.upper() == "SELL":
            if pos["units"] < quantity:
                raise ValueError("Insufficient position to sell in paper trade")
            pos["units"] -= quantity
            self.state["cash"] += cost
            if pos["units"] == 0:
                pos["avg_price"] = 0.0
        else:
            raise ValueError("side must be BUY or SELL")

        self.state["positions"][symbol] = pos
        self.state["orders"].append({
            "id": order_id, "symbol": symbol, "side": side.upper(),
            "qty": quantity, "price": price, "timestamp": time.time()
        })
        self._save()
        log.info(f"[PAPER] {side.upper()} {quantity} {symbol} @ {price} (order_id={order_id})")
        return order_id

# ----------------------------
# E*TRADE Broker (OAuth 1.0a)
# ----------------------------
class ETradeBroker(Broker):
    """
    Minimal example of signed OAuth 1.0a requests for E*TRADE.
    You must complete the OAuth dance beforehand to get oauth_token & oauth_token_secret.

    Docs (high level):
      - Sandbox base: https://apisb.etrade.com
      - Prod base:    https://api.etrade.com
      - Orders:       /v1/accounts/{accountIdKey}/orders/place.json
      - Accounts:     /v1/accounts/list.json
      - Balance:      /v1/accounts/{accountIdKey}/balance.json

    NOTE: You may need additional headers, proper JSON/XML bodies, and entitlements.
    """
    def __init__(self, cfg: Config, account_id_key: Optional[str] = None):
        self.cfg = cfg
        self.base = cfg.etrade_base
        self.consumer_key = cfg.consumer_key
        self.consumer_secret = cfg.consumer_secret
        self.oauth_token = cfg.oauth_token
        self.oauth_token_secret = cfg.oauth_token_secret
        self.account_id_key = account_id_key or self._pick_default_account()

    # ---------------- OAuth helpers ----------------
    def _nonce(self, n=8) -> str:
        return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(n))

    def _timestamp(self) -> str:
        return str(int(time.time()))

    def _percent_encode(self, s: str) -> str:
        return up.quote(s, safe="~")

    def _sign(self, method: str, url: str, params: Dict[str, str]) -> str:
        # OAuth 1.0a signature base string
        param_str = "&".join(f"{self._percent_encode(k)}={self._percent_encode(params[k])}"
                             for k in sorted(params.keys()))
        base_str = "&".join([
            method.upper(),
            self._percent_encode(url),
            self._percent_encode(param_str)
        ])
        key = f"{self._percent_encode(self.consumer_secret)}&{self._percent_encode(self.oauth_token_secret)}"
        digest = hmac.new(key.encode(), base_str.encode(), hashlib.sha1).digest()
        return base64.b64encode(digest).decode()

    def _auth_header(self, method: str, url: str, extra_params: Dict[str, str]) -> str:
        oauth_params = {
            "oauth_consumer_key": self.consumer_key,
            "oauth_nonce": self._nonce(),
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": self._timestamp(),
            "oauth_token": self.oauth_token,
            "oauth_version": "1.0",
        }
        all_params = {**oauth_params, **extra_params}
        oauth_signature = self._sign(method, url, all_params)
        oauth_params["oauth_signature"] = oauth_signature
        header = "OAuth " + ", ".join(f'{k}="{self._percent_encode(v)}"' for k, v in oauth_params.items())
        return header

    # ---------------- API calls (minimal) ----------------
    def _pick_default_account(self) -> str:
        url = f"{self.base}/v1/accounts/list.json"
        hdr = {"Authorization": self._auth_header("GET", url, {})}
        r = requests.get(url, headers=hdr, timeout=10)
        r.raise_for_status()
        data = r.json()
        # TODO: confirm response shape; this is a placeholder selection
        account_id_key = data["AccountListResponse"]["Accounts"]["Account"][0]["accountIdKey"]
        log.info(f"Using account_id_key={account_id_key}")
        return account_id_key

    def get_equity(self) -> float:
        url = f"{self.base}/v1/accounts/{self.account_id_key}/balance.json"
        hdr = {"Authorization": self._auth_header("GET", url, {})}
        r = requests.get(url, headers=hdr, timeout=10)
        r.raise_for_status()
        data = r.json()
        # TODO: verify field names per environment
        equity = float(data["BalanceResponse"]["accountBalance"]["totalAccountValue"])
        return equity

    def get_position(self, symbol: str) -> Tuple[int, Optional[float]]:
        url = f"{self.base}/v1/accounts/{self.account_id_key}/portfolio.json"
        hdr = {"Authorization": self._auth_header("GET", url, {})}
        r = requests.get(url, headers=hdr, timeout=10)
        r.raise_for_status()
        data = r.json()
        positions = data["PortfolioResponse"]["AccountPortfolio"][0].get("Position", [])
        qty, avg = 0, None
        for p in positions:
            # Symbols may appear under different fields; adjust as needed
            if p.get("symbolDescription") == symbol or p.get("Product", {}).get("symbol") == symbol:
                qty = int(p["quantity"])
                avg = float(p.get("pricePaid", 0.0)) if p.get("pricePaid") else None
                break
        return qty, avg

    def place_order(self, symbol: str, side: str, quantity: int, price: Optional[float] = None) -> str:
        """
        Places a market order. For limit orders, add price & orderTerm.
        E*TRADE accepts JSON or XML payloads depending on endpoint configuration.
        """
        url = f"{self.base}/v1/accounts/{self.account_id_key}/orders/place.json"
        payload = {
            "PlaceOrderRequest": {
                "orderType": "EQ",
                "clientOrderId": f"dlbot-{int(time.time())}",
                "Order": [{
                    "allOrNone": False,
                    "priceType": "MARKET",
                    "orderTerm": "GOOD_FOR_DAY",
                    "marketSession": "REGULAR",
                    "Instrument": [{
                        "Product": {"symbol": symbol, "securityType": "EQ"},
                        "orderAction": side.upper(),  # "BUY" or "SELL"
                        "quantityType": "QUANTITY",
                        "quantity": quantity
                    }]
                }]
            }
        }
        hdr = {
            "Authorization": self._auth_header("POST", url, {}),
            "Content-Type": "application/json"
        }
        r = requests.post(url, headers=hdr, json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        order_id = str(data["PlaceOrderResponse"]["orderId"])
        log.info(f"[LIVE] {side.upper()} {quantity} {symbol} (order_id={order_id})")
        return order_id

# ----------------------------
# Risk management & sizing
# ----------------------------
def compute_units(cfg: Config, broker: Broker, symbol: str, price: float) -> int:
    equity = broker.get_equity()
    max_by_value = int(cfg.max_position_value // price)
    max_by_pct = int((equity * cfg.max_position_pct) // price)
    base = min(cfg.units, max_by_value, max_by_pct)
    return max(0, base)

def should_exit(cfg: Config, entry: Optional[float], price: float) -> Optional[str]:
    if entry is None:
        return None
    if price <= entry * (1 - cfg.stop_loss_pct):
        return "STOP_LOSS"
    if price >= entry * (1 + cfg.take_profit_pct):
        return "TAKE_PROFIT"
    return None

# ----------------------------
# Main loop
# ----------------------------
def run_bot(cfg: Config):
    cfg.validate()
    data = DataFeed()
    strat = DLStrategy(cfg.model_path)

    broker: Broker = PaperBroker(cfg.paper_ledger_path) if cfg.mode == "paper" else ETradeBroker(cfg)

    # price history per symbol
    price_hist: Dict[str, List[float]] = {s: [] for s in cfg.symbols}

    log.info(f"Starting bot in {cfg.mode.upper()} mode for {', '.join(cfg.symbols)}")
    log.info(f"E*TRADE base URL: {cfg.etrade_base}")
    while True:
        try:
            quotes = data.get_quotes(cfg.symbols)  # dict: symbol -> price

            for sym in cfg.symbols:
                px = quotes[sym]
                price_hist[sym].append(px)
                if len(price_hist[sym]) < 32:
                    # warmup per symbol
                    continue

                feats = data.build_features(price_hist[sym])
                action, conf = strat.predict_action(feats)
                units, avg = broker.get_position(sym)
                avg_price = avg if avg is not None else None
                log.info(f"{sym} | Price={px:.2f} | Pos={units} @ {avg_price} | Action={action} (p={conf:.2f})")

                # exits
                exit_reason = should_exit(cfg, avg_price, px)
                if units != 0 and exit_reason:
                    oid = broker.close_position(sym)
                    log.info(f"{sym} | Exit due to {exit_reason}; order_id={oid}")
                    continue

                # entries
                if action == "BUY":
                    target_qty = compute_units(cfg, broker, sym, px)
                    if target_qty > 0 and units == 0:
                        fill_price = px if cfg.mode == "paper" else None  # live is MARKET
                        oid = broker.place_order(sym, "BUY", target_qty, price=fill_price)
                        log.info(f"{sym} | Opened LONG {target_qty} (order_id={oid})")
                elif action == "SELL":
                    if units > 0:
                        fill_price = px if cfg.mode == "paper" else None
                        oid = broker.place_order(sym, "SELL", units, price=fill_price)
                        log.info(f"{sym} | Closed LONG {units} (order_id={oid})")

            time.sleep(cfg.poll_seconds)

        except KeyboardInterrupt:
            log.info("Shutting down...")
            break
        except Exception as e:
            log.exception(f"Error in main loop: {e}")
            time.sleep(cfg.poll_seconds)

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Deep-Learning Stock Trading Bot")
    p.add_argument("--mode", choices=["paper", "live"], default=os.getenv("BOT_MODE", "paper"))
    p.add_argument("--symbols", nargs="+", default=os.getenv("BOT_SYMBOLS", "AAPL MSFT").split())
    p.add_argument("--model-path", default=os.getenv("BOT_MODEL_PATH", "./models/model.pt"))
    p.add_argument("--poll-seconds", type=int, default=int(os.getenv("BOT_POLL", "30")))
    p.add_argument("--units", type=int, default=int(os.getenv("BOT_UNITS", "5")))
    p.add_argument("--max-position-value", type=float, default=float(os.getenv("BOT_MAX_POS_VALUE", "10000")))
    p.add_argument("--max-position-pct", type=float, default=float(os.getenv("BOT_MAX_POS_PCT", "0.25")))
    p.add_argument("--stop-loss-pct", type=float, default=float(os.getenv("BOT_SL", "0.02")))
    p.add_argument("--take-profit-pct", type=float, default=float(os.getenv("BOT_TP", "0.04")))
    args = p.parse_args()

    cfg = Config(
        mode=args.mode,
        symbols=args.symbols,
        model_path=args.model_path,
        poll_seconds=args.poll_seconds,
        units=args.units,
        max_position_value=args.max_position_value,
        max_position_pct=args.max_position_pct,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
    )
    run_bot(cfg)
