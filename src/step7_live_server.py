"""
=============================================================
Phase 4 | Step 7: FastAPI Live Server (Production Quant Grade)
=============================================================
Major Upgrades Addressing Comprehensive Mentor Critique:
  1. 100% Stationary Feature Alignment:
     - Exactly mirrors step4: % distances, returns, rolling OBV z-score.
     - Zero raw dollar price features passed to model.
  2. True 2-Candle (10-Minute) Horizon Scoring Queue:
     - Each signal is held in pending status for 2 candles (10 min).
     - Scored against the exact +/- 0.08% flat threshold.
  3. Dynamic Model-Driven Scenarios:
     - Target distances scale with model confidence (higher P = wider target).
     - Explanations dynamically adapt to real RSI, MACD, and volume values.
  4. Disk Persistence:
     - Logs saved to logs/prediction_log.json; survives server restarts.
  5. Anti-Repainting Guarantee:
     - Only closed candles evaluated; logged predictions are immutable.
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
LOG_FILE   = "logs/prediction_log.json"
os.makedirs("logs", exist_ok=True)

# ── Logging setup ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("TSLA-Server")

# ═══════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════
class AppState:
    def __init__(self):
        self.engine          = None          # model package dict
        self.candle_buffer   = pd.DataFrame()# rolling window of recent candles
        self.prediction_log  = []            # all predictions ever made
        self.pending_queue   = []            # predictions waiting for 2nd candle (10m)
        self.latest_pred     = None          # most recent prediction dict
        self.connected_ws    = []            # active WebSocket clients
        self.server_start    = datetime.now(timezone.utc)
        self.total_scored    = 0
        self.correct_scored  = 0
        self.last_poll_time  = None

STATE = AppState()

def load_saved_log():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                saved = json.load(f)
                for p in saved:
                    if p.get('signal') == 'NEUTRAL':
                        p['correct'] = None
                        status_str = str(p.get('status', ''))
                        clean_diff = re.sub(r'[❌✅🟡⏳\(\)]', '', status_str).replace('Broke ', '').replace('HOLD', '').replace('FLAT', '').strip()
                        p['status'] = f"HOLD ({clean_diff})" if clean_diff else "HOLD"
                STATE.prediction_log = saved
                # Recompute scores strictly on directional signals (UP / DOWN)
                scored = [p for p in saved if p.get('correct') is not None and p.get('signal') in ['UP', 'DOWN']]
                STATE.total_scored = len(scored)
                STATE.correct_scored = sum(1 for p in scored if p.get('correct') is True)
                log.info(f"Loaded {len(saved)} persistent log entries ({STATE.correct_scored}/{STATE.total_scored} directional wins)")
        except Exception as e:
            log.warning(f"Could not load log file: {e}")

def save_log_to_disk():
    try:
        with open(LOG_FILE, "w") as f:
            json.dump(STATE.prediction_log, f, indent=2, default=str)
    except Exception as e:
        log.error(f"Error saving log to disk: {e}")

# ═══════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════
def load_engine():
    if not Path(MODEL_PATH).exists():
        log.error(f"Model not found at {MODEL_PATH}. Run step5 first.")
        sys.exit(1)
    pkg = joblib.load(MODEL_PATH)
    log.info(f"Model loaded ✓ ({len(pkg['feature_cols'])} stationary features, Threshold: UP>={pkg['confidence_high']})")
    return pkg

# ═══════════════════════════════════════════════════════════
# ALPACA DATA FETCHER
# ═══════════════════════════════════════════════════════════
def fetch_recent_bars(n_bars: int = 100) -> pd.DataFrame:
    """Fetch recent closed 5-minute TSLA bars from Alpaca."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed

    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=5)

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

    # Timezone conversion to America/New_York
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")

    # Filter out current unclosed bar
    now_ny = datetime.now(timezone.utc).astimezone(df.index.tz)
    df = df[df.index < now_ny - timedelta(minutes=5)]
    return df.tail(n_bars)

# ═══════════════════════════════════════════════════════════
# STATIONARY FEATURE ENGINEERING (Mirrors step4 exactly)
# ═══════════════════════════════════════════════════════════
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close  = df['Close']
    high   = df['High']
    low    = df['Low']
    open_p = df['Open']
    volume = df['Volume']

    # 1. Trend
    df['rsi_14'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df['macd_pct']        = (macd_obj.macd() / close) * 100.0
    df['macd_signal_pct'] = (macd_obj.macd_signal() / close) * 100.0
    df['macd_hist_pct']   = (macd_obj.macd_diff() / close) * 100.0

    ema_9  = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    ema_21 = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    df['price_vs_ema_9']  = (close - ema_9) / ema_9 * 100.0
    df['price_vs_ema_21'] = (close - ema_21) / ema_21 * 100.0
    df['ema_cross_pct']   = (ema_9 - ema_21) / ema_21 * 100.0

    # 2. Volatility
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_mid   = bb.bollinger_mavg()
    df['price_vs_bb_upper'] = (close - bb_upper) / close * 100.0
    df['price_vs_bb_lower'] = (close - bb_lower) / close * 100.0
    df['price_vs_bb_mid']   = (close - bb_mid) / bb_mid * 100.0
    df['bb_pct']            = bb.bollinger_pband()
    df['bb_width']          = bb.bollinger_wband()
    atr_14 = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df['atr_pct']           = (atr_14 / close) * 100.0
    df['raw_atr']           = atr_14  # kept for scenario dollar bands

    # 3. Momentum
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['roc_5']   = ta.momentum.ROCIndicator(close, window=5).roc()
    df['roc_10']  = ta.momentum.ROCIndicator(close, window=10).roc()

    # 4. Volume
    vol_sma_10      = volume.rolling(10).mean()
    df['vol_ratio'] = volume / (vol_sma_10 + 1e-8)
    raw_obv  = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    obv_mean = raw_obv.rolling(20).mean()
    obv_std  = raw_obv.rolling(20).std() + 1e-8
    df['obv_zscore'] = (raw_obv - obv_mean) / obv_std

    typical_price = (high + low + close) / 3.0
    vwap_approx   = (typical_price * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-8)
    df['price_vs_vwap'] = (close - vwap_approx) / vwap_approx * 100.0

    # 5. Time
    df['hour']           = df.index.hour + df.index.minute / 60.0
    df['is_first_30min'] = ((df.index.hour == 9)  & (df.index.minute >= 30)).astype(int)
    df['is_last_30min']  = ((df.index.hour == 15) & (df.index.minute >= 30)).astype(int)
    df['is_power_hour']  = (df.index.hour == 15).astype(int)

    # 6. Price Action
    df['price_vs_high_20'] = (close - high.rolling(20).max()) / close * 100.0
    df['price_vs_low_20']  = (close - low.rolling(20).min())  / close * 100.0
    df['candle_body']      = (close - open_p) / open_p * 100.0
    df['upper_shadow']     = (high - np.maximum(open_p, close)) / close * 100.0
    df['lower_shadow']     = (np.minimum(open_p, close) - low) / close * 100.0

    df['ret_lag_1'] = close.pct_change(1) * 100.0
    df['ret_lag_2'] = close.pct_change(2) * 100.0
    df['ret_lag_3'] = close.pct_change(3) * 100.0
    df['rsi_lag_1'] = df['rsi_14'].shift(1)
    df['rsi_lag_2'] = df['rsi_14'].shift(2)
    df['vol_ratio_lag_1'] = df['vol_ratio'].shift(1)

    return df

# ═══════════════════════════════════════════════════════════
# MAKE ONE PREDICTION ON LATEST CLOSED CANDLE
# ═══════════════════════════════════════════════════════════
def make_prediction(engine: dict, df_with_features: pd.DataFrame) -> Optional[dict]:
    feature_cols = engine['feature_cols']
    model        = engine['model']
    conf_high    = engine['confidence_high']
    conf_low     = engine['confidence_low']

    latest = df_with_features.dropna(subset=feature_cols).tail(1)
    if latest.empty:
        return None

    row   = latest.iloc[-1]
    ts    = latest.index[-1]
    X     = row[feature_cols].values.reshape(1, -1)

    prob_up = float(model.predict_proba(X)[0, 1])

    if prob_up >= conf_high:
        direction = "UP"
        confidence = prob_up * 100.0
    elif prob_up <= conf_low:
        direction = "DOWN"
        confidence = (1.0 - prob_up) * 100.0
    else:
        direction = "NEUTRAL"
        confidence = max(prob_up, 1.0 - prob_up) * 100.0

    current_price = float(row['Close'])
    atr           = float(row.get('raw_atr', 1.5))
    vol_ratio     = float(row.get('vol_ratio', 1.0))
    rsi           = float(row.get('rsi_14', 50.0))
    macd_hist     = float(row.get('macd_hist_pct', 0.0))
    stoch_k       = float(row.get('stoch_k', 50.0))
    bb_pct        = float(row.get('bb_pct', 0.5))

    # Model-driven targets: scale base target with model confidence
    conviction_mult = 0.8 + max(0.0, (confidence - 50.0) / 100.0)
    base_move = round(atr * conviction_mult, 2)
    bull_move = round(atr * 1.5, 2)
    bear_move = round(atr * 1.2, 2)

    vol_note = "Volume is normal"
    if vol_ratio > 2.0:
        bull_move = round(bull_move * 1.3, 2)
        bear_move = round(bear_move * 1.2, 2)
        vol_note = f"Volume spike ({vol_ratio:.1f}x avg) — breakout potential elevated"
    elif vol_ratio < 0.5:
        base_move = round(base_move * 0.7, 2)
        vol_note = f"Low volume ({vol_ratio:.1f}x avg) — narrow consolidation range"

    # Dynamic explanation checking actual indicators
    if rsi < 35:
        rsi_desc = f"RSI={rsi:.0f} (oversold bounce setup)"
    elif rsi > 65:
        rsi_desc = f"RSI={rsi:.0f} (overbought momentum stretch)"
    else:
        rsi_desc = f"RSI={rsi:.0f} (neutral zone)"

    macd_desc = "MACD accelerating" if macd_hist > 0 else "MACD decelerating"

    if direction == "UP":
        base_target = round(current_price + base_move, 2)
        bull_target = round(current_price + bull_move, 2)
        bear_target = round(current_price - bear_move, 2)
        base_expl = f"Model projects TSLA gains +${base_move:.2f} over 10m based on {rsi_desc} and {macd_desc}."
        bull_expl = f"Optimistic expansion: ${bull_target:.2f} (+${bull_move:.2f}) if buyer volume continues."
        bear_expl = f"Risk/Invalidation level: ${bear_target:.2f} (-${bear_move:.2f}). Hard stop-loss zone."
    elif direction == "DOWN":
        base_target = round(current_price - base_move, 2)
        bull_target = round(current_price + bear_move, 2)
        bear_target = round(current_price - bull_move, 2)
        base_expl = f"Model projects TSLA pulls back -${base_move:.2f} over 10m. {rsi_desc}, {macd_desc}."
        bull_expl = f"Upside invalidation: ${bull_target:.2f} (+${bear_move:.2f}) if buyers absorb sell pressure."
        bear_expl = f"Downside momentum target: ${bear_target:.2f} (-${bull_move:.2f}) if support breaks."
    else:
        base_target = round(current_price, 2)
        bull_target = round(current_price + base_move, 2)
        bear_target = round(current_price - base_move, 2)
        base_expl = f"Confidence ({confidence:.1f}%) below trigger threshold. TSLA likely consolidates near ${current_price:.2f}."
        bull_expl = f"Resistance ceiling: ${bull_target:.2f}."
        bear_expl = f"Support floor: ${bear_target:.2f}."

    prediction = {
        "timestamp"     : str(ts)[:19],
        "ticker"        : TICKER,
        "signal"        : direction,
        "confidence"    : round(confidence, 1),
        "prob_up"       : round(prob_up * 100.0, 1),
        "price"         : round(current_price, 2),
        "open"          : round(float(row['Open']), 2),
        "high"          : round(float(row['High']), 2),
        "low"           : round(float(row['Low']), 2),
        "volume"        : int(row['Volume']),
        "rsi"           : round(rsi, 1),
        "atr"           : round(atr, 2),
        "vol_ratio"     : round(vol_ratio, 2),
        "macd_hist"     : round(macd_hist, 4),
        "stoch_k"       : round(stoch_k, 1),
        "bb_pct"        : round(bb_pct * 100.0, 1),
        "ema_cross"     : round(float(row.get('ema_cross_pct', 0)), 3),
        "base_target"   : base_target,
        "bull_target"   : bull_target,
        "bear_target"   : bear_target,
        "base_expl"     : base_expl,
        "bull_expl"     : bull_expl,
        "bear_expl"     : bear_expl,
        "vol_note"      : vol_note,
        "actual_price"  : None,
        "correct"       : None,
        "status"        : "Pending (10m)",
        "logged_at"     : datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "id"            : len(STATE.prediction_log) + 1,
        "candles_passed": 0
    }
    return prediction

# ═══════════════════════════════════════════════════════════
# 2-CANDLE (10-MIN) HORIZON SCORING ENGINE
# ═══════════════════════════════════════════════════════════
def update_pending_queue(new_candle_price: float):
    """
    Every time a new 5-minute candle closes:
    Increments elapsed candles for all pending predictions.
    When a prediction reaches exactly 2 candles (10 minutes),
    it is scored using the exact +/- 0.08% threshold!
    """
    to_remove = []
    for pred in STATE.pending_queue:
        pred['candles_passed'] += 1
        if pred['candles_passed'] >= 2:
            # 10 minutes have elapsed! Evaluate outcome:
            entry_price = pred['price']
            pct_move = (new_candle_price - entry_price) / entry_price * 100.0
            pred['actual_price'] = round(new_candle_price, 2)
            dollar_diff = round(new_candle_price - entry_price, 2)
            diff_str = f"+${dollar_diff:.2f}" if dollar_diff >= 0 else f"-${abs(dollar_diff):.2f}"

            if pred['signal'] == "UP":
                if pct_move > 0.08:
                    pred['correct'] = True
                    pred['status'] = f"WIN ({diff_str})"
                elif pct_move < -0.08:
                    pred['correct'] = False
                    pred['status'] = f"LOSS ({diff_str})"
                else:
                    pred['correct'] = None  # Flat / Breakeven
                    pred['status'] = f"FLAT ({diff_str})"

            elif pred['signal'] == "DOWN":
                if pct_move < -0.08:
                    pred['correct'] = True
                    pred['status'] = f"WIN ({diff_str})"
                elif pct_move > 0.08:
                    pred['correct'] = False
                    pred['status'] = f"LOSS ({diff_str})"
                else:
                    pred['correct'] = None
                    pred['status'] = f"FLAT ({diff_str})"

            else: # NEUTRAL (Abstain / Hold - not a directional trade, zero dollars at risk)
                pred['correct'] = None
                pred['status']  = f"HOLD ({diff_str})"

            if pred['correct'] is not None:
                STATE.total_scored += 1
                if pred['correct'] is True:
                    STATE.correct_scored += 1

            to_remove.append(pred)
            log.info(f"Signal #{pred['id']} resolved: {pred['status']}")

    for p in to_remove:
        STATE.pending_queue.remove(p)

    save_log_to_disk()

# ═══════════════════════════════════════════════════════════
# BACKGROUND POLLING LOOP
# ═══════════════════════════════════════════════════════════
async def polling_loop():
    log.info("Background polling task started.")
    await asyncio.sleep(3)

    while True:
        try:
            df_raw = await asyncio.get_event_loop().run_in_executor(
                None, lambda: fetch_recent_bars(100)
            )
            STATE.candle_buffer = df_raw
            STATE.last_poll_time = datetime.now(timezone.utc)

            df_feat = compute_features(df_raw)
            pred = make_prediction(STATE.engine, df_feat)

            if pred:
                last_ts = STATE.prediction_log[-1]['timestamp'] if STATE.prediction_log else None
                is_new_candle = (pred['timestamp'] != last_ts)

                if is_new_candle:
                    # 1. Update pending predictions using this newly closed candle price
                    update_pending_queue(pred['price'])

                    # 2. Add current prediction to persistent log and pending queue
                    STATE.prediction_log.append(pred)
                    STATE.pending_queue.append(pred)
                    STATE.latest_pred = pred
                    save_log_to_disk()
                    log.info(f"NEW candle logged: {pred['timestamp']} @ ${pred['price']} | Signal: {pred['signal']}")
                else:
                    STATE.latest_pred = pred

                # Broadcast updates
                await broadcast({"type": "prediction", "data": pred})
                await broadcast({"type": "log_update", "data": get_log_summary()})

        except Exception as e:
            log.error(f"Polling error: {e}")
            await broadcast({"type": "error", "message": str(e)})

        await asyncio.sleep(60)

async def broadcast(message: dict):
    dead = []
    for ws in STATE.connected_ws:
        try:
            await ws.send_text(json.dumps(message, default=str))
        except Exception:
            dead.append(ws)
    for ws in dead:
        STATE.connected_ws.remove(ws)

def get_log_summary() -> list:
    entries = []
    for p in STATE.prediction_log[-50:]:
        entries.append({
            "id"           : p['id'],
            "time"         : p['timestamp'][-8:],
            "signal"       : p['signal'],
            "confidence"   : f"{p['confidence']:.1f}%",
            "price"        : f"${p['price']:.2f}",
            "rsi"          : f"{p['rsi']:.0f}",
            "base_target"  : f"${p['base_target']:.2f}",
            "bull_target"  : f"${p['bull_target']:.2f}",
            "bear_target"  : f"${p['bear_target']:.2f}",
            "actual"       : f"${p['actual_price']:.2f}" if p['actual_price'] else "—",
            "correct"      : p.get('status', '⏳'),
            "vol_note"     : p['vol_note'],
            "logged_at"    : p['logged_at'],
        })
    return list(reversed(entries))

# ═══════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=" * 55)
    log.info("  TSLA AI INDICATOR SERVER STARTING")
    log.info("=" * 55)
    load_saved_log()
    STATE.engine = load_engine()
    task = asyncio.create_task(polling_loop())
    log.info("Server ready → http://localhost:8000")
    yield
    task.cancel()
    save_log_to_disk()
    log.info("Server shut down cleanly.")

app = FastAPI(title="TSLA AI Indicator", version="2.0", lifespan=lifespan)

@app.get("/api/status")
async def get_status():
    uptime = datetime.now(timezone.utc) - STATE.server_start
    acc = (STATE.correct_scored / STATE.total_scored * 100.0
           if STATE.total_scored > 0 else None)
    return JSONResponse({
        "status"          : "running",
        "ticker"          : TICKER,
        "uptime_seconds"  : int(uptime.total_seconds()),
        "total_predictions": len(STATE.prediction_log),
        "signals_scored"  : STATE.total_scored,
        "correct_signals" : STATE.correct_scored,
        "live_accuracy"   : f"{acc:.1f}%" if acc else "N/A (waiting for 10m horizon)",
        "last_poll"       : str(STATE.last_poll_time)[:19] if STATE.last_poll_time else "Not yet",
        "connected_clients": len(STATE.connected_ws),
    })

@app.get("/api/latest")
async def get_latest():
    if not STATE.latest_pred:
        return JSONResponse({"error": "No predictions yet. Market may be closed."}, status_code=404)
    return JSONResponse(STATE.latest_pred)

@app.get("/api/log")
async def get_log():
    return JSONResponse({"predictions": get_log_summary()})

@app.get("/api/candles")
async def get_candles():
    if STATE.candle_buffer.empty:
        return JSONResponse({"candles": []})
    df = STATE.candle_buffer.tail(80).copy()
    candles = []
    for ts, row in df.iterrows():
        candles.append({
            "time"  : str(ts),
            "open"  : round(row['Open'],  2),
            "high"  : round(row['High'],  2),
            "low"   : round(row['Low'],   2),
            "close" : round(row['Close'], 2),
            "volume": int(row['Volume']),
        })
    return JSONResponse({"candles": candles})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    STATE.connected_ws.append(websocket)
    if STATE.latest_pred:
        await websocket.send_text(json.dumps({"type": "prediction", "data": STATE.latest_pred}, default=str))
    await websocket.send_text(json.dumps({"type": "log_update", "data": get_log_summary()}, default=str))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        STATE.connected_ws.remove(websocket)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = Path("src/dashboard.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h2>Dashboard not found.</h2>")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="warning")
