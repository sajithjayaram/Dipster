from fastapi import FastAPI, APIRouter, HTTPException, Query, Depends, Header, Request
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Literal, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import os
import logging
from pathlib import Path
import uuid

# Third-party services
import asyncio

try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None

import pandas as pd
import numpy as np
import yfinance as yf
import httpx
import json
from zoneinfo import ZoneInfo

# Auth deps
from passlib.context import CryptContext
from jose import jwt, JWTError
from starlette.responses import RedirectResponse, JSONResponse

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'test_database')]

# App & Router
app = FastAPI(title="AI Trading Agent API", version="0.8.0")
api_router = APIRouter(prefix="/api")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"]
)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("trading-ai")

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-not-for-prod")
JWT_ALGO = os.environ.get("JWT_ALGO", "HS256")
JWT_EXPIRE_MIN = int(os.environ.get("JWT_EXPIRE_MIN", "10080"))

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# ---------------- Models (India-only) ----------------
class AnalyzeRequest(BaseModel):
    symbol: str
    timeframe: Literal['weekly', 'daily', 'intraday'] = 'weekly'
    market: Literal['IN'] = 'IN'
    source: Literal['yahoo', 'msn'] = 'yahoo'

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

class AITightRecommendation(BaseModel):
    symbol: str
    timeframe: str
    action: Literal['buy', 'sell', 'hold']
    confidence: float = Field(ge=0, le=100)
    reasons: List[str]
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class StrategyFilters(BaseModel):
    risk_tolerance: Literal['low', 'medium', 'high'] = 'medium'
    horizon: Literal['intraday', 'daily', 'weekly', 'longterm'] = 'weekly'
    asset_classes: List[Literal['stocks']] = Field(default_factory=lambda: ['stocks'])
    market: Literal['IN'] = 'IN'
    momentum_preference: bool = True
    value_preference: bool = False
    rsi_min: Optional[int] = None
    rsi_max: Optional[int] = None
    sectors: Optional[List[Literal['IT','Banking','Auto','Pharma','FMCG','Energy','Metals']]] = None
    allocation: Dict[str, int] = Field(default_factory=lambda: {"stocks": 100})
    caps_allocation: Dict[Literal['largecap','midcap','smallcap'], int] = Field(default_factory=lambda: {"largecap": 60, "midcap": 25, "smallcap": 15})

class StrategyRequest(BaseModel):
    filters: StrategyFilters
    prompt: Optional[str] = None
    top_n: int = 5
    source: Literal['yahoo','msn'] = 'yahoo'

class StrategyPick(BaseModel):
    symbol: str
    name: Optional[str] = None
    asset_class: Literal['stocks'] = 'stocks'
    sector: Optional[str] = None
    cap: Optional[Literal['largecap','midcap','smallcap']] = None
    score: float
    action: Literal['buy','sell','hold']
    reasons: List[str]

# ---------------- Data load ----------------
INDIA_TICKERS: List[Dict[str,str]] = []
INDIA_SECTORS: Dict[str, Dict[str, str]] = {}
try:
    with open(ROOT_DIR/"data"/"india_tickers.json","r") as f:
        INDIA_TICKERS = json.load(f)
except Exception:
    INDIA_TICKERS = []
try:
    with open(ROOT_DIR/"data"/"india_sectors.json","r") as f:
        for row in json.load(f):
            INDIA_SECTORS[row['symbol']] = {"sector": row.get('sector'), "cap": row.get('cap'), "name": row.get('name')}
except Exception:
    INDIA_SECTORS = {}

# ---------------- Market Data & Indicators ----------------
async def fetch_ohlcv_yahoo(symbol: str, timeframe: str) -> pd.DataFrame:
    if timeframe == 'weekly': interval = '1wk'; period = '5y'
    elif timeframe == 'daily': interval = '1d'; period = '2y'
    else: interval = '60m'; period = '60d'
    data = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if data is None or data.empty:
        raise ValueError("No data returned from Yahoo Finance")
    return data.dropna().rename(columns=str.title)

async def fetch_ohlcv_msn(symbol: str, timeframe: str) -> pd.DataFrame:
    return await fetch_ohlcv_yahoo(symbol, timeframe)

async def fetch_ohlcv(symbol: str, timeframe: str, source: str = 'yahoo') -> pd.DataFrame:
    if source == 'msn':
        return await fetch_ohlcv_msn(symbol, timeframe)
    return await fetch_ohlcv_yahoo(symbol, timeframe)


def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    close = df['Close']
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors='coerce')
    sma_50 = close.rolling(50).mean(); sma_200 = close.rolling(200).mean()
    delta = close.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    roll_up = gain.rolling(14).mean(); roll_down = loss.rolling(14).mean(); rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean(); ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26; macd_signal = macd_line.ewm(span=9, adjust=False).mean(); macd_hist = macd_line - macd_signal
    def _num(x):
        try:
            if isinstance(x, (pd.Series, pd.DataFrame)): x = x.iloc[-1] if hasattr(x,'iloc') else x.squeeze()
            if pd.isna(x): return None
            return float(x)
        except Exception: return None
    last = df.index[-1]
    snapshot = {
        'price': _num(close.iloc[-1]), 'sma_50': _num(sma_50.iloc[-1]), 'sma_200': _num(sma_200.iloc[-1]),
        'rsi_14': _num(rsi.iloc[-1]), 'macd': _num(macd_line.iloc[-1]), 'macd_signal': _num(macd_signal.iloc[-1]),
        'macd_hist': _num(macd_hist.iloc[-1]), 'date': last.isoformat() if hasattr(last,'isoformat') else str(last)
    }
    heur_action = 'hold'; reasons = []; score = 50.0
    if snapshot['rsi_14'] is not None:
        if snapshot['rsi_14'] < 30: reasons.append("RSI<30 oversold"); heur_action='buy'; score+=15
        elif snapshot['rsi_14'] > 70: reasons.append("RSI>70 overbought"); heur_action='sell'; score-=15
    if snapshot['sma_50'] and snapshot['sma_200']:
        if snapshot['sma_50'] > snapshot['sma_200']: reasons.append("SMA50>SMA200 uptrend"); score+=20; heur_action = heur_action if heur_action!='hold' else 'buy'
        elif snapshot['sma_50'] < snapshot['sma_200']: reasons.append("SMA50<SMA200 downtrend"); score-=20; heur_action = heur_action if heur_action!='hold' else 'sell'
    if snapshot['macd_hist'] is not None:
        if snapshot['macd_hist'] > 0: reasons.append("MACD hist positive"); score+=10
        else: reasons.append("MACD hist negative"); score-=10
    score = max(0.0, min(100.0, score))
    return {'snapshot': snapshot, 'baseline_signal': heur_action, 'baseline_reasons': reasons, 'baseline_confidence': score}

# ---------------- Strategy helpers ----------------
async def build_universe(filters: StrategyFilters, limit: int = 80) -> List[Dict[str,str]]:
    # India-only: start from sector mapping if available, fallback to tickers list
    out: List[Dict[str,str]] = []
    if filters.sectors:
        # use INDIA_SECTORS to gather all matching sectors
        wanted = set(filters.sectors)
        for sym, meta in INDIA_SECTORS.items():
            if meta.get('sector') in wanted:
                out.append({"symbol": sym, "name": meta.get('name'), "sector": meta.get('sector'), "cap": meta.get('cap')})
    else:
        # general list limited
        for it in INDIA_TICKERS[:50]:
            sym = it.get('symbol');
            meta = INDIA_SECTORS.get(sym, {})
            out.append({"symbol": sym, "name": it.get('name'), "sector": meta.get('sector'), "cap": meta.get('cap')})
    # Dedup and cap
    seen = set(); dedup = []
    for it in out:
        sym = it.get('symbol');
        if not sym or sym in seen: continue
        seen.add(sym); dedup.append(it)
    return dedup[:limit]

async def score_symbol(sym: str, timeframe: str, source: str, momentum: bool, value: bool) -> Dict[str, Any]:
    try:
        df = await fetch_ohlcv(sym, timeframe, source)
        ind = compute_indicators(df)
        score = ind.get('baseline_confidence', 60.0)
        snap = ind['snapshot']
        if momentum:
            if (snap.get('macd_hist') or 0) > 0: score += 5
            if snap.get('sma_50') and snap.get('sma_200') and snap['sma_50'] > snap['sma_200']: score += 5
        if value:
            if snap.get('sma_50') and snap.get('price') and snap['price'] < snap['sma_50']: score += 5
        score = max(0.0, min(100.0, score))
        return {"ok": True, "ind": ind, "score": score}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def proportional_take(total: int, pct: int, min_each: int = 1) -> int:
    return max(min_each, round((pct/100.0)*total))

# ---------------- Routes ----------------
@api_router.post("/strategy/suggest")
async def strategy_suggest(req: StrategyRequest,
                           llm_key: Optional[str] = Header(default=None, alias='X-LLM-KEY'),
                           llm_provider: Optional[str] = Header(default='openai', alias='X-LLM-PROVIDER'),
                           llm_model: Optional[str] = Header(default=None, alias='X-LLM-MODEL')):
    # Build universe with optional sectors
    universe = await build_universe(req.filters, limit=80)
    results: List[Dict[str, Any]] = []
    horizon = req.filters.horizon if req.filters.horizon in ['daily','weekly'] else 'weekly'
    # Evaluate up to 60
    for it in universe[:60]:
        sym = it.get('symbol')
        r = await score_symbol(sym, horizon, req.source, req.filters.momentum_preference, req.filters.value_preference)
        if r.get('ok'):
            results.append({
                "symbol": sym,
                "name": it.get('name'),
                "sector": it.get('sector'),
                "cap": it.get('cap') or 'largecap',
                "score": r['score'],
                "ind": r['ind'],
                "asset_class": 'stocks'
            })
    # RSI filter
    if req.filters.rsi_min is not None or req.filters.rsi_max is not None:
        min_r = req.filters.rsi_min if req.filters.rsi_min is not None else -999
        max_r = req.filters.rsi_max if req.filters.rsi_max is not None else 999
        def ok_rsi(x):
            rsi = x['ind']['snapshot'].get('rsi_14')
            return (rsi is None) or (min_r <= rsi <= max_r)
        results = [x for x in results if ok_rsi(x)]

    # Within stocks: apply caps allocation (large/mid/small)
    total_take = max(1, min(req.top_n, 10))
    caps = req.filters.caps_allocation
    buckets = { 'largecap': [], 'midcap': [], 'smallcap': [] }
    for x in results:
        buckets[x.get('cap','largecap')].append(x)
    for k in buckets:
        buckets[k].sort(key=lambda z: z['score'], reverse=True)
    take_large = proportional_take(total_take, caps.get('largecap', 60))
    take_mid = proportional_take(total_take, caps.get('midcap', 25))
    take_small = proportional_take(total_take, caps.get('smallcap', 15))
    selected = buckets['largecap'][:take_large] + buckets['midcap'][:take_mid] + buckets['smallcap'][:take_small]
    # If we selected too many, trim by score
    if len(selected) > total_take:
        selected.sort(key=lambda z: z['score'], reverse=True)
        selected = selected[:total_take]
    # If too few, top up from remaining pool
    if len(selected) < total_take:
        remaining = [x for x in sorted(results, key=lambda z: z['score'], reverse=True) if x not in selected]
        selected += remaining[: total_take - len(selected)]

    # Build picks
    picks: List[StrategyPick] = []
    for x in selected:
        ind = x['ind']
        action = ind.get('baseline_signal','hold')
        reasons = ind.get('baseline_reasons', [])
        picks.append(StrategyPick(symbol=x['symbol'], name=x.get('name'), asset_class='stocks', sector=x.get('sector'), cap=x.get('cap'), score=float(x['score']), action=action, reasons=reasons))

    used_ai = False
    if llm_key and llm_provider == 'openai' and AsyncOpenAI:
        try:
            client_oa = AsyncOpenAI(api_key=llm_key)
            summary = [{"symbol": p.symbol, "score": p.score, "sector": p.sector, "cap": p.cap} for p in picks]
            prompt = f"User filters: {req.filters.model_dump()}\nUser prompt: {req.prompt or ''}\nCandidates: {summary}\nReturn concise reasons per symbol as JSON array of objects {{symbol, short_reason}}."
            resp = await client_oa.responses.create(model=llm_model or os.environ.get('OPENAI_MODEL','gpt-5-mini'), input=[{"role":"user","content":prompt}], max_output_tokens=300)
            txt = resp.output_text or ""
            if txt:
                data = json.loads(txt)
                for p in picks:
                    found = next((d for d in data if d.get('symbol')==p.symbol), None)
                    if found and found.get('short_reason'):
                        p.reasons = [found['short_reason']]
                used_ai = True
        except Exception as e:
            logger.info("Strategy AI refinement skipped: %s", e)

    return {"picks": [p.model_dump() for p in picks], "used_ai": used_ai, "note": "India stocks-only, sector-aware, caps allocation"}

# Health
@api_router.get("/")
async def root():
    return {"message": "Trading AI backend is up"}

app.include_router(api_router)
