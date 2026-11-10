"""
=============================================================
Phase 4 | Step 7: FastAPI Live Server
=============================================================
What this server does:
  1. On startup: loads the prediction engine + fetches recent
     TSLA history to warm up the indicator buffer
  2. Background task: polls Alpaca every 60 seconds for the
     latest CLOSED 5-minute TSLA candle
  3. For each new candle: computes all 35 features, runs the
     model, generates 3 scenarios, logs the prediction
  4. Serves a REST API + WebSocket so the frontend gets
     live updates instantly

Anti-repainting guarantee:
  - Predictions are ONLY made on CLOSED candles
  - Once logged, a prediction is NEVER changed
  - Each log entry is timestamped at the moment of prediction

Run with:
    python src/step7_live_server.py

Then open:
    http://localhost:8000
=============================================================
"""

import os, sys, json, asyncio, logging, joblib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np

# ── FastAPI ──────────────────────────────────────────────
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ── Technical Indicators ─────────────────────────────────
import ta

# ── Alpaca ───────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
TICKER     = os.getenv("TICKER", "TSLA")

MODEL_PATH = "models/tsla_xgb_model.pkl"

# ── Logging setup ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("TSLA-Server")

# ═══════════════════════════════════════════════════════════
# GLOBAL STATE
# (In production use a database; for demo in-memory is fine)
# ═══════════════════════════════════════════════════════════
class AppState:
    def __init__(self):
        self.engine          = None          # prediction engine package
        self.candle_buffer   = pd.DataFrame()# rolling window of recent candles
        self.prediction_log  = []            # all predictions ever made
        self.latest_pred     = None          # most recent prediction dict
        self.connected_ws    = []            # active WebSocket clients
        self.server_start    = datetime.now(timezone.utc)
        self.total_signals   = 0
        self.correct_signals = 0
        self.last_poll_time  = None
        self.market_open     = False

STATE = AppState()


# ═══════════════════════════════════════════════════════════
# LOAD MODEL (from tsla_xgb_model.pkl — no function objects stored)
# ═══════════════════════════════════════════════════════════
def load_engine():
    if not Path(MODEL_PATH).exists():
        log.error(f"Model not found at {MODEL_PATH}. Run step5 first.")
        sys.exit(1)
    pkg = joblib.load(MODEL_PATH)
    log.info(f"Model loaded ✓  ({len(pkg['feature_cols'])} features)")
    return pkg


# ═══════════════════════════════════════════════════════════
# ALPACA DATA FETCHER
# ═══════════════════════════════════════════════════════════
def fetch_recent_bars(n_bars: int = 100) -> pd.DataFrame:
    """Fetch the last n_bars of 5-minute TSLA bars from Alpaca REST API."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed

    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=5)   # fetch last 5 days to ensure enough bars

    req = StockBarsRequest(
        symbol_or_symbols=TICKER,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=DataFeed.IEX
    )
    bars = client.get_stock_bars(req)
    df = bars.df

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)

    df = df[['open','high','low','close','volume']].copy()
    df.columns = ['Open','High','Low','Close','Volume']
    df = df.sort_index()

    # Remove the current (still-open) candle — only use CLOSED candles
    now_utc = datetime.now(timezone.utc)
    df = df[df.index < now_utc - timedelta(minutes=5)]

    return df.tail(n_bars)


# ═══════════════════════════════════════════════════════════
# FEATURE ENGINEERING (mirrors step4 exactly)
# CRITICAL: must match training pipeline — no differences allowed
# ═══════════════════════════════════════════════════════════
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 35 technical features on a candle buffer."""
    df = df.copy()
    close  = df['Close']
    high   = df['High']
    low    = df['Low']
    volume = df['Volume']

    # Trend
    df['rsi_14']     = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd_obj         = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df['macd']       = macd_obj.macd()
    df['macd_signal']= macd_obj.macd_signal()
    df['macd_hist']  = macd_obj.macd_diff()
    df['ema_9']      = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    df['ema_21']     = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    df['ema_cross']  = df['ema_9'] - df['ema_21']

    # Volatility
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df['bb_upper']   = bb.bollinger_hband()
    df['bb_lower']   = bb.bollinger_lband()
    df['bb_mid']     = bb.bollinger_mavg()
    df['bb_pct']     = bb.bollinger_pband()
    df['bb_width']   = bb.bollinger_wband()
    df['atr_14']     = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # Momentum
    stoch            = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df['stoch_k']    = stoch.stoch()
    df['stoch_d']    = stoch.stoch_signal()
    df['roc_5']      = ta.momentum.ROCIndicator(close, window=5).roc()
    df['roc_10']     = ta.momentum.ROCIndicator(close, window=10).roc()

    # Volume
    df['vol_sma_10'] = volume.rolling(10).mean()
    df['vol_ratio']  = volume / df['vol_sma_10']
    df['obv']        = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df['typical_price'] = (high + low + close) / 3
    df['vwap_approx']= (df['typical_price'] * volume).rolling(20).sum() / volume.rolling(20).sum()
    df['price_vs_vwap'] = (close - df['vwap_approx']) / df['vwap_approx'] * 100

    # Time
    df['hour']           = df.index.hour + df.index.minute / 60
    df['is_first_30min'] = ((df.index.hour == 9)  & (df.index.minute < 60)).astype(int)
    df['is_last_30min']  = ((df.index.hour == 15) & (df.index.minute >= 30)).astype(int)
    df['is_power_hour']  = (df.index.hour == 15).astype(int)

    # Price context
    df['price_vs_high_20'] = (close - high.rolling(20).max()) / close * 100
    df['price_vs_low_20']  = (close - low.rolling(20).min())  / close * 100
    df['candle_body']      = (df['Close'] - df['Open']) / df['Open'] * 100

    # Lags
    for lag in [1, 2, 3]:
        df[f'close_lag_{lag}'] = close.shift(lag)
        df[f'rsi_lag_{lag}']   = df['rsi_14'].shift(lag)

    return df


# ═══════════════════════════════════════════════════════════
# MAKE ONE PREDICTION on the latest closed candle
# ═══════════════════════════════════════════════════════════
def make_prediction(engine: dict, df_with_features: pd.DataFrame) -> Optional[dict]:
    """Run the model on the most recent completed candle."""
    feature_cols = engine['feature_cols']
    model        = engine['model']
    conf_high    = engine['confidence_high']
    conf_low     = engine['confidence_low']

    # Use the last row (most recent closed candle)
    latest = df_with_features.dropna(subset=feature_cols).tail(1)
    if latest.empty:
        log.warning("Not enough history to compute features yet.")
        return None

    row   = latest.iloc[-1]
    ts    = latest.index[-1]
    X     = row[feature_cols].values.reshape(1, -1)

    prob_up = float(model.predict_proba(X)[0, 1])

    # Signal direction
    if prob_up >= conf_high:
        direction = "UP"
        confidence = prob_up * 100
    elif prob_up <= conf_low:
        direction = "DOWN"
        confidence = (1 - prob_up) * 100
    else:
        direction = "NEUTRAL"
        confidence = max(prob_up, 1 - prob_up) * 100

    # Scenario computation
    current_price = float(row['Close'])
    atr           = float(row.get('atr_14', 2.0))
    vol_ratio     = float(row.get('vol_ratio', 1.0))
    rsi           = float(row.get('rsi_14', 50.0))
    macd_hist     = float(row.get('macd_hist', 0.0))
    bb_pct        = float(row.get('bb_pct', 0.5))

    base_move = atr * 0.8
    bull_move = atr * 1.5
    bear_move = atr * 1.2

    vol_note = "Volume is normal"
    if vol_ratio > 2.0:
        bull_move *= 1.3
        bear_move *= 1.2
        vol_note = f"⚡ High volume spike ({vol_ratio:.1f}x avg) — extended move possible"
    elif vol_ratio < 0.5:
        base_move *= 0.6
        bull_move *= 0.6
        vol_note = f"🔇 Low volume ({vol_ratio:.1f}x avg) — move may be muted"

    if direction == "UP":
        base_target = round(current_price + base_move, 2)
        bull_target = round(current_price + bull_move, 2)
        bear_target = round(current_price - bear_move, 2)
        base_expl = f"Model predicts TSLA rises ~${base_move:.2f} in next 10 min. RSI={rsi:.0f} (oversold → recovery). MACD hist={macd_hist:.3f}."
        bull_expl = f"If volume stays elevated, price could reach ${bull_target:.2f} (+${bull_move:.2f}). Watch for RSI crossing above 50."
        bear_expl = f"If prediction fails (e.g. sudden market sell-off), invalidation level is ${bear_target:.2f}. Stop-loss zone."
    elif direction == "DOWN":
        base_target = round(current_price - base_move, 2)
        bull_target = round(current_price + bear_move, 2)
        bear_target = round(current_price - bull_move, 2)
        base_expl = f"Model predicts TSLA drops ~${base_move:.2f} in next 10 min. RSI={rsi:.0f}, MACD hist={macd_hist:.3f} (weakening)."
        bull_expl = f"If buyers step in and reverse the move, recovery to ${bull_target:.2f} (+${bear_move:.2f}) is possible."
        bear_expl = f"If selling pressure accelerates, extended downside to ${bear_target:.2f} (-${bull_move:.2f}). Watch volume."
    else:
        base_target = round(current_price, 2)
        bull_target = round(current_price + base_move, 2)
        bear_target = round(current_price - base_move, 2)
        base_expl = f"Model confidence too low for a directional call. TSLA likely consolidates near ${current_price:.2f}."
        bull_expl = f"Upside scenario: ${bull_target:.2f} if a catalyst emerges (news, volume spike)."
        bear_expl = f"Downside support: ${bear_target:.2f}. Watch for breakdown below this level."

    prediction = {
        # Core signal
        "timestamp"     : str(ts)[:19],
        "ticker"        : TICKER,
        "signal"        : direction,
        "confidence"    : round(confidence, 1),
        "prob_up"       : round(prob_up * 100, 1),

        # Current candle data
        "price"         : round(current_price, 2),
        "open"          : round(float(row['Open']), 2),
        "high"          : round(float(row['High']), 2),
        "low"           : round(float(row['Low']), 2),
        "volume"        : int(row['Volume']),

        # Supporting indicators
        "rsi"           : round(rsi, 1),
        "atr"           : round(atr, 2),
        "vol_ratio"     : round(vol_ratio, 2),
        "macd_hist"     : round(macd_hist, 4),
        "bb_pct"        : round(bb_pct * 100, 1),
        "ema_cross"     : round(float(row.get('ema_cross', 0)), 3),

        # Scenarios
        "base_target"   : base_target,
        "bull_target"   : bull_target,
        "bear_target"   : bear_target,
        "base_move"     : round(base_move, 2),
        "base_expl"     : base_expl,
        "bull_expl"     : bull_expl,
        "bear_expl"     : bear_expl,
        "vol_note"      : vol_note,

        # Log table fields (filled in later)
        "actual_price"  : None,
        "correct"       : None,
        "logged_at"     : datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "id"            : len(STATE.prediction_log) + 1,
    }
    return prediction


# ═══════════════════════════════════════════════════════════
# BACKGROUND POLLING TASK
# ═══════════════════════════════════════════════════════════
async def polling_loop():
    """
    Runs every 60 seconds. Fetches the latest TSLA 5-min bar,
    computes features, makes a prediction, and broadcasts to
    all connected WebSocket clients.
    """
    log.info("Background polling task started.")
    await asyncio.sleep(5)   # brief startup delay

    while True:
        try:
            log.info("Fetching latest TSLA bars from Alpaca...")
            df_raw = await asyncio.get_event_loop().run_in_executor(
                None, lambda: fetch_recent_bars(100)
            )
            STATE.candle_buffer = df_raw
            STATE.last_poll_time = datetime.now(timezone.utc)

            df_feat = compute_features(df_raw)
            pred = make_prediction(STATE.engine, df_feat)

            if pred:
                # Update the previous prediction's "actual_price" if enough time has passed
                if STATE.prediction_log:
                    prev = STATE.prediction_log[-1]
                    if prev['actual_price'] is None:
                        prev['actual_price'] = pred['price']
                        if prev['signal'] == "UP":
                            prev['correct'] = pred['price'] > prev['price']
                        elif prev['signal'] == "DOWN":
                            prev['correct'] = pred['price'] < prev['price']
                        else:
                            prev['correct'] = None
                        if prev['correct'] is not None:
                            STATE.total_signals += 1
                            if prev['correct']:
                                STATE.correct_signals += 1

                STATE.prediction_log.append(pred)
                STATE.latest_pred = pred

                sig_icon = {"UP":"🟢","DOWN":"🔴","NEUTRAL":"🟡"}.get(pred['signal'],'⚪')
                log.info(
                    f"{sig_icon} {pred['signal']:<7} "
                    f"Conf={pred['confidence']:.1f}%  "
                    f"Price=${pred['price']:.2f}  "
                    f"Base→${pred['base_target']:.2f}  "
                    f"RSI={pred['rsi']:.0f}"
                )

                # Broadcast to all connected frontend clients
                await broadcast({"type": "prediction", "data": pred})
                await broadcast({"type": "log_update", "data": get_log_summary()})

        except Exception as e:
            log.error(f"Polling error: {e}")
            await broadcast({"type": "error", "message": str(e)})

        await asyncio.sleep(60)   # wait 60 seconds before next poll


async def broadcast(message: dict):
    """Send a JSON message to all connected WebSocket clients."""
    dead = []
    for ws in STATE.connected_ws:
        try:
            await ws.send_text(json.dumps(message, default=str))
        except Exception:
            dead.append(ws)
    for ws in dead:
        STATE.connected_ws.remove(ws)


def get_log_summary() -> list:
    """Return last 50 log entries formatted for the UI table."""
    entries = []
    for p in STATE.prediction_log[-50:]:
        entries.append({
            "id"           : p['id'],
            "time"         : p['timestamp'][-8:],   # HH:MM:SS
            "signal"       : p['signal'],
            "confidence"   : f"{p['confidence']:.1f}%",
            "price"        : f"${p['price']:.2f}",
            "base_target"  : f"${p['base_target']:.2f}",
            "bull_target"  : f"${p['bull_target']:.2f}",
            "bear_target"  : f"${p['bear_target']:.2f}",
            "actual"       : f"${p['actual_price']:.2f}" if p['actual_price'] else "—",
            "correct"      : ("✅" if p['correct'] else "❌") if p['correct'] is not None else "⏳",
            "vol_note"     : p['vol_note'],
            "rsi"          : p['rsi'],
            "logged_at"    : p['logged_at'],
        })
    return list(reversed(entries))   # newest first


# ═══════════════════════════════════════════════════════════
# FASTAPI APP  (using modern lifespan handler — no deprecation warning)
# ═══════════════════════════════════════════════════════════
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────
    log.info("=" * 55)
    log.info("  TSLA AI INDICATOR SERVER STARTING")
    log.info("=" * 55)
    STATE.engine = load_engine()
    task = asyncio.create_task(polling_loop())
    log.info("Server ready → http://localhost:8000")
    yield
    # ── Shutdown ─────────────────────────────────────────
    task.cancel()
    log.info("Server shut down cleanly.")

app = FastAPI(title="TSLA AI Indicator", version="1.0", lifespan=lifespan)


# ── REST Endpoints ───────────────────────────────────────

@app.get("/api/status")
async def get_status():
    """Server health check and stats."""
    uptime = datetime.now(timezone.utc) - STATE.server_start
    acc = (STATE.correct_signals / STATE.total_signals * 100
           if STATE.total_signals > 0 else None)
    return JSONResponse({
        "status"          : "running",
        "ticker"          : TICKER,
        "uptime_seconds"  : int(uptime.total_seconds()),
        "total_predictions": len(STATE.prediction_log),
        "signals_scored"  : STATE.total_signals,
        "correct_signals" : STATE.correct_signals,
        "live_accuracy"   : f"{acc:.1f}%" if acc else "N/A (not enough data yet)",
        "last_poll"       : str(STATE.last_poll_time)[:19] if STATE.last_poll_time else "Not yet",
        "connected_clients": len(STATE.connected_ws),
    })


@app.get("/api/latest")
async def get_latest():
    """Return the most recent prediction."""
    if not STATE.latest_pred:
        return JSONResponse({"error": "No predictions yet. Market may be closed."}, status_code=404)
    return JSONResponse(STATE.latest_pred)


@app.get("/api/log")
async def get_log():
    """Return the full prediction log table."""
    return JSONResponse({"predictions": get_log_summary()})


@app.get("/api/candles")
async def get_candles():
    """Return recent OHLCV candles for the chart."""
    if STATE.candle_buffer.empty:
        return JSONResponse({"candles": []})
    df = STATE.candle_buffer.tail(80).copy()
    df.index = df.index.astype(str)
    candles = []
    for ts, row in df.iterrows():
        candles.append({
            "time"  : ts[:19],
            "open"  : round(row['Open'],  2),
            "high"  : round(row['High'],  2),
            "low"   : round(row['Low'],   2),
            "close" : round(row['Close'], 2),
            "volume": int(row['Volume']),
        })
    return JSONResponse({"candles": candles})


# ── WebSocket ────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time WebSocket connection for live frontend updates."""
    await websocket.accept()
    STATE.connected_ws.append(websocket)
    client_ip = websocket.client.host
    log.info(f"WebSocket client connected: {client_ip}  (total: {len(STATE.connected_ws)})")

    # Send current state immediately on connect
    if STATE.latest_pred:
        await websocket.send_text(json.dumps(
            {"type": "prediction", "data": STATE.latest_pred}, default=str))
    await websocket.send_text(json.dumps(
        {"type": "log_update", "data": get_log_summary()}, default=str))

    try:
        while True:
            await websocket.receive_text()   # keep connection alive
    except WebSocketDisconnect:
        STATE.connected_ws.remove(websocket)
        log.info(f"WebSocket client disconnected: {client_ip}")


# ── Dashboard HTML (served at /) ─────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard HTML page."""
    html_path = Path("src/dashboard.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h2>Dashboard not found. Place dashboard.html in src/</h2>")


# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    log.info("Starting TSLA AI Indicator server...")
    uvicorn.run(
        app,                   # pass the app object directly (not a string)
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="warning",   # suppress uvicorn noise — our logger handles output
    )
