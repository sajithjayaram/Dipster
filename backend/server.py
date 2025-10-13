from fastapi import FastAPI, APIRouter, HTTPException, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Literal, Optional, Dict, Any
from datetime import datetime, timezone
import os
import logging
from pathlib import Path
import uuid

# Third-party services
import asyncio

try:
    # OpenAI official SDK (2025) – Responses API
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # Will validate at runtime

import pandas as pd
import numpy as np
import yfinance as yf

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection (available for future features like alert persistence)
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'test_database')]

# FastAPI app and router (/api prefix is mandatory for ingress)
app = FastAPI(title="AI Trading Agent API", version="0.1.0")
api_router = APIRouter(prefix="/api")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("trading-ai")

# --------------- Models ---------------
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class AnalyzeRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol (e.g., RELIANCE.NS, TCS.NS, INFY.NS)")
    timeframe: Literal['weekly', 'daily', 'intraday'] = Field('weekly', description="Analysis timeframe")
    market: Literal['IN', 'US', 'OTHER'] = Field('IN', description="Market/region hint")

class AIRecommendation(BaseModel):
    symbol: str
    timeframe: str
    action: Literal['buy', 'sell', 'hold']
    confidence: float = Field(ge=0, le=100)
    reasons: List[str]
    indicators_snapshot: Dict[str, Any]
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Strict model for Structured Outputs (no free-form dicts)
class AITightRecommendation(BaseModel):
    symbol: str
    timeframe: str
    action: Literal['buy', 'sell', 'hold']
    confidence: float = Field(ge=0, le=100)
    reasons: List[str]
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

# --------------- Utilities ---------------

def _safely_get_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val

async def fetch_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    # Use Yahoo Finance via yfinance
    # Map timeframe to yfinance interval
    if timeframe == 'weekly':
        interval = '1wk'
        period = '5y'
    elif timeframe == 'daily':
        interval = '1d'
        period = '2y'
    else:
        # intraday – use 60m for broader coverage
        interval = '60m'
        period = '60d'

    try:
        data = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
        if data is None or data.empty:
            raise ValueError("No data returned from Yahoo Finance")
        data = data.dropna().rename(columns=str.title)  # Ensure 'Close', 'Open', etc.
        return data
    except Exception as e:
        logger.exception("Failed to fetch data for %s", symbol)
        raise HTTPException(status_code=400, detail=f"Failed to fetch market data: {e}")


def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    close = df['Close']
    # Ensure close is a 1D float Series
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors='coerce')

    # Simple Moving Averages
    sma_50 = close.rolling(window=50).mean()
    sma_200 = close.rolling(window=200).mean()

    # RSI (14) – pure pandas to avoid ndarray shape issues
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(14).mean()
    roll_down = loss.rolling(14).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # MACD (12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    # Last row snapshot
    last = df.index[-1]
    # Use iloc to avoid potential ambiguity with index
    v_close = close.iloc[-1]
    v50 = sma_50.iloc[-1]
    v200 = sma_200.iloc[-1]
    v_rsi = rsi.iloc[-1]
    v_macd = macd_line.iloc[-1]
    v_sig = macd_signal.iloc[-1]
    v_hist = macd_hist.iloc[-1]

    def _num(x):
        try:
            if isinstance(x, (pd.Series, pd.DataFrame)):
                # Reduce to scalar
                x = x.iloc[-1] if hasattr(x, 'iloc') else x.squeeze()
            if pd.isna(x):
                return None
            return float(x)
        except Exception:
            return None

    snapshot = {
        'price': _num(v_close),
        'sma_50': _num(v50),
        'sma_200': _num(v200),
        'rsi_14': _num(v_rsi),
        'macd': _num(v_macd),
        'macd_signal': _num(v_sig),
        'macd_hist': _num(v_hist),
        'date': last.isoformat() if hasattr(last, 'isoformat') else str(last)
    }

    # Simple heuristics for a baseline signal (used for guardrail/fallback)
    heur_action = 'hold'
    reasons = []
    if snapshot['rsi_14'] is not None:
        if snapshot['rsi_14'] < 30:
            reasons.append("RSI indicates oversold (<30)")
            heur_action = 'buy'
        elif snapshot['rsi_14'] > 70:
            reasons.append("RSI indicates overbought (>70)")
            heur_action = 'sell'
    if snapshot['sma_50'] and snapshot['sma_200']:
        if snapshot['sma_50'] > snapshot['sma_200']:
            reasons.append("SMA50 above SMA200 (uptrend)")
            if heur_action == 'hold':
                heur_action = 'buy'
        elif snapshot['sma_50'] < snapshot['sma_200']:
            reasons.append("SMA50 below SMA200 (downtrend)")
            if heur_action == 'hold':
                heur_action = 'sell'
    if snapshot['macd_hist'] is not None:
        if snapshot['macd_hist'] > 0:
            reasons.append("MACD histogram positive")
        else:
            reasons.append("MACD histogram negative")

    return {
        'snapshot': snapshot,
        'baseline_signal': heur_action,
        'baseline_reasons': reasons,
    }


async def call_openai_recommendation(symbol: str, timeframe: str, indicators: Dict[str, Any]) -> AIRecommendation:
    if AsyncOpenAI is None:
        raise HTTPException(status_code=500, detail="OpenAI SDK not available on server")

    api_key = _safely_get_env('OPENAI_API_KEY')
    client_oa = AsyncOpenAI(api_key=api_key)

    # Use Responses API with structured output
    system_instruction = (
        "You are a cautious, senior trading analyst. Analyze provided technical indicators to produce a single, actionable recommendation "
        "for the specified timeframe. Prefer conservative, risk-aware guidance. Keep output concise."
    )

    user_content = (
        f"Symbol: {symbol}\nTimeframe: {timeframe}\n"
        f"Indicators (latest snapshot): {indicators['snapshot']}\n"
        f"Baseline signal (heuristic): {indicators['baseline_signal']} | reasons: {indicators['baseline_reasons']}\n"
        "Return only the structured recommendation."
    )

    # Guard for long requests – enforce timeout
    try:
        # New Structured Outputs via responses.parse
        resp = await asyncio.wait_for(
            client_oa.responses.parse(
                model=os.environ.get('OPENAI_MODEL', 'gpt-5-mini'),
                input=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content},
                ],
                # Map to our STRICT Pydantic model for the LLM response
                text_format=AITightRecommendation,
                reasoning={"effort": os.environ.get('OPENAI_REASONING', 'low')},
                max_output_tokens=int(os.environ.get('OPENAI_MAX_TOKENS', '350')),
            ),
            timeout=float(os.environ.get('OPENAI_TIMEOUT_SECONDS', '30')),
        )
        tight = resp.output_parsed
        # Adapt to full schema by adding indicators snapshot and timestamp
        snap = indicators['snapshot']
        # Normalize confidence to 0-100 scale if model returned 0-1
        conf = float(tight.confidence)
        if conf <= 1.0:
            conf = round(conf * 100.0, 2)
        conf = max(0.0, min(100.0, conf))
        return AIRecommendation(
            symbol=tight.symbol,
            timeframe=tight.timeframe,
            action=tight.action,
            confidence=conf,
            reasons=tight.reasons,
            indicators_snapshot=snap,
            stop_loss=tight.stop_loss,
            take_profit=tight.take_profit,
        )
    except asyncio.TimeoutError:
        logger.error("OpenAI request timed out for %s", symbol)
        raise HTTPException(status_code=504, detail="AI analysis timed out. Please retry.")
    except Exception as e:
        logger.exception("OpenAI parse failed, falling back to heuristic: %s", e)
        # Fallback to baseline heuristics into our schema
        snap = indicators['snapshot']
        return AIRecommendation(
            symbol=symbol,
            timeframe=timeframe,
            action=indicators['baseline_signal'],
            confidence=60.0,
            reasons=["AI unavailable. Using baseline technical heuristics."] + indicators['baseline_reasons'],
            indicators_snapshot=snap,
            stop_loss=round(snap['price'] * (0.95 if indicators['baseline_signal'] == 'buy' else 1.05), 2) if snap.get('price') else None,
            take_profit=round(snap['price'] * (1.08 if indicators['baseline_signal'] == 'buy' else 0.92), 2) if snap.get('price') else None,
        )

# --------------- Routes ---------------

@api_router.get("/")
async def root():
    return {"message": "Trading AI backend is up"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_obj = StatusCheck(client_name=input.client_name)
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    items = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    for it in items:
        if isinstance(it.get('timestamp'), str):
            it['timestamp'] = datetime.fromisoformat(it['timestamp'])
    return items

@api_router.post("/analyze", response_model=AIRecommendation)
async def analyze(req: AnalyzeRequest):
    # Normalize Indian NSE symbols: user might omit ".NS"; do not over-assume – leave as given
    symbol = req.symbol.strip()
    # Fetch OHLCV and compute indicators
    df = await fetch_ohlcv(symbol, req.timeframe)
    indicators = compute_indicators(df)
    # Ensure we have enough data (SMA200 requires >= 200 periods)
    if len(df) < 210 and req.timeframe in ("daily", "weekly"):
        logger.info("Limited history for %s: %d rows", symbol, len(df))
    # Call OpenAI for AI-enhanced recommendation
    rec = await call_openai_recommendation(symbol, req.timeframe, indicators)
    # enrich with snapshot (parse may already include, but ensure)
    if not rec.indicators_snapshot:
        rec.indicators_snapshot = indicators['snapshot']
    return rec

@api_router.get("/signal/current", response_model=AIRecommendation)
async def current_signal(symbol: str = Query(..., description="Ticker symbol"), timeframe: Literal['weekly','daily','intraday'] = 'weekly'):
    df = await fetch_ohlcv(symbol.strip(), timeframe)
    indicators = compute_indicators(df)
    rec = await call_openai_recommendation(symbol.strip(), timeframe, indicators)
    if not rec.indicators_snapshot:
        rec.indicators_snapshot = indicators['snapshot']
    return rec

# Mount router
app.include_router(api_router)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
