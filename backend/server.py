from fastapi import FastAPI, APIRouter, HTTPException, Query, Header, Request
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, EmailStr
from typing import List, Literal, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import os
import logging
from pathlib import Path
import uuid
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
from zoneinfo import ZoneInfo  # noqa: F401 (kept for future use)
from urllib.parse import urlencode
from passlib.context import CryptContext
from jose import jwt, JWTError
from starlette.responses import RedirectResponse

# Env
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'test_database')]

# App
app = FastAPI(title="AI Trading Agent API", version="0.9.0")
api = APIRouter(prefix="/api")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("trading-ai")

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-not-for-prod")
JWT_ALGO = os.environ.get("JWT_ALGO", "HS256")
JWT_EXPIRE_MIN = int(os.environ.get("JWT_EXPIRE_MIN", "10080"))

# Google OAuth
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")  # noqa: F401

# ---------------- Models ----------------
class AnalyzeRequest(BaseModel):
    symbol: str
    timeframe: Literal['weekly', 'daily', 'intraday'] = 'weekly'
    market: Literal['IN'] = 'IN'
    source: Literal['yahoo', 'msn'] = 'yahoo'


class StrategyFilters(BaseModel):
    risk_tolerance: Literal['low', 'medium', 'high'] = 'medium'
    horizon: Literal['intraday', 'daily', 'weekly', 'longterm'] = 'weekly'
    asset_classes: List[Literal['stocks']] = Field(default_factory=lambda: ['stocks'])
    market: Literal['IN'] = 'IN'
    momentum_preference: bool = True
    value_preference: bool = False
    rsi_min: Optional[int] = None
    rsi_max: Optional[int] = None
    sectors: Optional[List[Literal['IT', 'Banking', 'Auto', 'Pharma', 'FMCG', 'Energy', 'Metals']]] = None
    allocation: Dict[str, int] = Field(default_factory=lambda: {"stocks": 100})
    caps_allocation: Dict[Literal['largecap', 'midcap', 'smallcap'], int] = Field(default_factory=lambda: {"largecap": 60, "midcap": 25, "smallcap": 15})


class Profile(BaseModel):
    provider: Literal['openai', 'anthropic', 'gemini'] = 'openai'
    model: Optional[str] = Field(default=os.environ.get('OPENAI_MODEL', 'gpt-5-mini'))


class UserOut(BaseModel):
    id: str
    email: EmailStr
    profile: Profile


class WatchlistUpdate(BaseModel):
    symbols: List[str]


class TelegramConfig(BaseModel):
    chat_id: Optional[str] = None
    enabled: bool = False
    buy_threshold: int = 80
    sell_threshold: int = 60
    frequency_min: int = 60
    quiet_start_hour: int = 22
    quiet_end_hour: int = 7
    timezone: str = 'Asia/Kolkata'


class SymbolThreshold(BaseModel):
    symbol: str
    buy_threshold: int
    sell_threshold: int


class SymbolThresholdsPayload(BaseModel):
    items: List[SymbolThreshold]


class StrategyRequest(BaseModel):
    filters: StrategyFilters
    prompt: Optional[str] = None
    top_n: int = 5
    source: Literal['yahoo', 'msn'] = 'yahoo'


class StrategyPick(BaseModel):
    symbol: str
    name: Optional[str] = None
    asset_class: Literal['stocks'] = 'stocks'
    sector: Optional[str] = None
    cap: Optional[Literal['largecap', 'midcap', 'smallcap']] = None
    score: float
    action: Literal['buy', 'sell', 'hold']
    reasons: List[str]


# ---------------- Data load ----------------
INDIA_TICKERS: List[Dict[str, str]] = []
INDIA_SECTORS: Dict[str, Dict[str, str]] = {}
try:
    with open(ROOT_DIR / "data" / "india_tickers.json", "r") as f:
        INDIA_TICKERS = json.load(f)
except Exception:
    INDIA_TICKERS = []
try:
    with open(ROOT_DIR / "data" / "india_sectors.json", "r") as f:
        for row in json.load(f):
            INDIA_SECTORS[row['symbol']] = {
                "sector": row.get('sector'),
                "cap": row.get('cap'),
                "name": row.get('name'),
            }
except Exception:
    INDIA_SECTORS = {}


# ---------------- Market Data & Indicators ----------------
async def fetch_ohlcv_yahoo(symbol: str, timeframe: str) -> pd.DataFrame:
    if timeframe == 'weekly':
        interval = '1wk'
        period = '5y'
    elif timeframe == 'daily':
        interval = '1d'
        period = '2y'
    else:
        interval = '60m'
        period = '60d'
    data = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if data is None or data.empty:
        raise ValueError("No data returned from Yahoo Finance")
    return data.dropna().rename(columns=str.title)


async def fetch_ohlcv_msn(symbol: str, timeframe: str) -> pd.DataFrame:
    # Fallback to Yahoo for now
    return await fetch_ohlcv_yahoo(symbol, timeframe)


async def fetch_ohlcv(symbol: str, timeframe: str, source: str = 'yahoo') -> pd.DataFrame:
    if source == 'msn':
        return await fetch_ohlcv_msn(symbol, timeframe)
    return await fetch_ohlcv_yahoo(symbol, timeframe)


def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors='coerce')

    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(14).mean()
    roll_down = loss.rolling(14).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    def _num(x):
        try:
            if isinstance(x, (pd.Series, pd.DataFrame)):
                x = x.iloc[-1] if hasattr(x, 'iloc') else x.squeeze()
            if pd.isna(x):
                return None
            return float(x)
        except Exception:
            return None

    last = df.index[-1]
    snapshot = {
        'price': _num(close.iloc[-1]),
        'sma_50': _num(sma_50.iloc[-1]),
        'sma_200': _num(sma_200.iloc[-1]),
        'rsi_14': _num(rsi.iloc[-1]),
        'macd': _num(macd_line.iloc[-1]),
        'macd_signal': _num(macd_signal.iloc[-1]),
        'macd_hist': _num(macd_hist.iloc[-1]),
        'date': last.isoformat() if hasattr(last, 'isoformat') else str(last),
    }

    heur_action = 'hold'
    reasons: List[str] = []
    score = 50.0

    if snapshot['rsi_14'] is not None:
        if snapshot['rsi_14'] < 30:
            reasons.append("RSI<30 oversold")
            heur_action = 'buy'
            score += 15
        elif snapshot['rsi_14'] > 70:
            reasons.append("RSI>70 overbought")
            heur_action = 'sell'
            score -= 15

    if snapshot['sma_50'] and snapshot['sma_200']:
        if snapshot['sma_50'] > snapshot['sma_200']:
            reasons.append("SMA50>SMA200 uptrend")
            score += 20
            if heur_action == 'hold':
                heur_action = 'buy'
        elif snapshot['sma_50'] < snapshot['sma_200']:
            reasons.append("SMA50<SMA200 downtrend")
            score -= 20
            if heur_action == 'hold':
                heur_action = 'sell'

    if snapshot['macd_hist'] is not None:
        if snapshot['macd_hist'] > 0:
            reasons.append("MACD hist positive")
            score += 10
        else:
            reasons.append("MACD hist negative")
            score -= 10

    score = max(0.0, min(100.0, score))
    return {
        'snapshot': snapshot,
        'baseline_signal': heur_action,
        'baseline_reasons': reasons,
        'baseline_confidence': score,
    }


# ---------------- Strategy helpers ----------------
async def build_universe(filters: StrategyFilters, limit: int = 80) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if filters.sectors:
        wanted = set(filters.sectors)
        for sym, meta in INDIA_SECTORS.items():
            if meta.get('sector') in wanted:
                out.append({
                    "symbol": sym,
                    "name": meta.get('name'),
                    "sector": meta.get('sector'),
                    "cap": meta.get('cap'),
                })
    else:
        for it in INDIA_TICKERS[:50]:
            sym = it.get('symbol')
            meta = INDIA_SECTORS.get(sym, {})
            out.append({
                "symbol": sym,
                "name": it.get('name'),
                "sector": meta.get('sector'),
                "cap": meta.get('cap'),
            })
    # Dedup and limit
    seen = set()
    dedup: List[Dict[str, str]] = []
    for it in out:
        sym = it.get('symbol')
        if not sym or sym in seen:
            continue
        seen.add(sym)
        dedup.append(it)
    return dedup[:limit]


def create_token(uid: str, email: str) -> str:
    payload = {
        "sub": uid,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MIN),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


# ---------------- Auth helpers ----------------
async def _require_user(request: Request) -> Dict[str, Any]:
    auth = request.headers.get('Authorization')
    if not auth or not auth.lower().startswith('bearer '):
        raise HTTPException(status_code=401, detail="missing_or_invalid_token")
    token = auth.split(' ', 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        uid = payload.get('sub')
        if not uid:
            raise HTTPException(status_code=401, detail="invalid_token")
        user = await db.users.find_one({"id": uid})
        if not user:
            raise HTTPException(status_code=404, detail="user_not_found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="invalid_token")


# ---------------- Auth routes ----------------
@api.post("/auth/signup")
async def signup(req: Dict[str, Any]):
    email = req.get('email')
    password = req.get('password')
    if not email or not password:
        raise HTTPException(status_code=400, detail="missing_credentials")
    existing = await db.users.find_one({"email": email})
    if existing:
        raise HTTPException(status_code=400, detail="email_in_use")
    user = {
        "id": str(uuid.uuid4()),
        "email": email,
        "password_hash": pwd_context.hash(password),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": {"provider": "openai", "model": os.environ.get('OPENAI_MODEL', 'gpt-5-mini')},
        "watchlist": ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        "alert_settings": {"enabled": False, "buy_threshold": 80, "sell_threshold": 60, "frequency_min": 60, "quiet_start_hour": 22, "quiet_end_hour": 7, "timezone": 'Asia/Kolkata'},
        "symbol_thresholds": {},
    }
    await db.users.insert_one(user)
    token = create_token(user['id'], email)
    return {"access_token": token, "token_type": "bearer"}


@api.post("/auth/login")
async def login(req: Dict[str, Any]):
    email = req.get('email')
    password = req.get('password')
    user = await db.users.find_one({"email": email})
    if not user or not pwd_context.verify(password, user.get('password_hash', '')):
        raise HTTPException(status_code=401, detail="invalid_credentials")
    token = create_token(user['id'], email)
    return {"access_token": token, "token_type": "bearer"}


@api.get("/auth/me")
async def auth_me(request: Request):
    user = await _require_user(request)
    prof = user.get('profile') or {"provider": "openai", "model": os.environ.get('OPENAI_MODEL', 'gpt-5-mini')}
    return {"id": user['id'], "email": user['email'], "profile": prof}


@api.get("/profile")
async def get_profile(request: Request):
    user = await _require_user(request)
    prof = user.get('profile') or {"provider": "openai", "model": os.environ.get('OPENAI_MODEL', 'gpt-5-mini')}
    return prof


@api.put("/profile")
async def put_profile(request: Request, p: Dict[str, Any]):
    user = await _require_user(request)
    await db.users.update_one({"id": user['id']}, {"$set": {"profile": p}})
    return p


# ---------------- Google OAuth ----------------
@api.get("/auth/google/login")
async def google_login():
    if not GOOGLE_CLIENT_ID or not GOOGLE_REDIRECT_URI:
        raise HTTPException(status_code=500, detail="google_oauth_not_configured")
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "online",
        "prompt": "consent",
    }
    url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    return RedirectResponse(url)


@api.get("/auth/google/callback")
async def google_callback(code: Optional[str] = None, error: Optional[str] = None):
    def fail(msg: str):
        logger.warning(f"google_oauth_error: {msg}")
        return RedirectResponse(url=f"/?oauth_error={msg}")

    if error:
        return fail(error)
    if not code:
        return fail("missing_code")
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET or not GOOGLE_REDIRECT_URI:
        return fail("google_oauth_not_configured")

    token_data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    async with httpx.AsyncClient(timeout=15) as hx:
        tr = await hx.post("https://oauth2.googleapis.com/token", data=token_data)
        if tr.status_code != 200:
            return fail("token_exchange_failed")
        token_json = tr.json()
        access_token = token_json.get("access_token")
        if not access_token:
            return fail("no_access_token")
        ur = await hx.get("https://www.googleapis.com/oauth2/v3/userinfo", headers={"Authorization": f"Bearer {access_token}"})
        if ur.status_code != 200:
            return fail("userinfo_failed")
        userinfo = ur.json()

    email = userinfo.get("email")
    if not email:
        return fail("email_not_found")

    existing = await db.users.find_one({"email": email})
    if not existing:
        user = {
            "id": str(uuid.uuid4()),
            "email": email,
            "password_hash": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "profile": {"provider": "openai", "model": os.environ.get('OPENAI_MODEL', 'gpt-5-mini')},
            "watchlist": ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
            "alert_settings": {"enabled": False, "buy_threshold": 80, "sell_threshold": 60, "frequency_min": 60, "quiet_start_hour": 22, "quiet_end_hour": 7, "timezone": 'Asia/Kolkata'},
            "symbol_thresholds": {},
        }
        await db.users.insert_one(user)
        uid = user['id']
    else:
        uid = existing['id']

    token = create_token(uid, email)
    return RedirectResponse(url=f"/?token={token}")


# ---------------- Search, Analyze, Signal ----------------
@api.get("/search")
async def search(q: str = Query(..., min_length=1), region: Optional[str] = Query('IN')):
    try:
        ql = q.strip().lower()
        results = []
        for it in INDIA_TICKERS:
            sym = it.get('symbol', '')
            name = it.get('name', '')
            if ql in sym.lower() or ql in name.lower():
                results.append({"symbol": sym, "name": name})
            if len(results) >= 15:
                break
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/analyze")
async def analyze(req: AnalyzeRequest,
                  llm_key: Optional[str] = Header(default=None, alias='X-LLM-KEY'),
                  llm_provider: Optional[str] = Header(default='openai', alias='X-LLM-PROVIDER'),
                  llm_model: Optional[str] = Header(default=None, alias='X-LLM-MODEL')):
    try:
        df = await fetch_ohlcv(req.symbol, req.timeframe, req.source)
        ind = compute_indicators(df)
        snapshot = ind['snapshot']
        action = ind['baseline_signal']
        reasons = ind['baseline_reasons']
        confidence = ind['baseline_confidence']
        return {
            "symbol": req.symbol,
            "timeframe": req.timeframe,
            "action": action,
            "confidence": confidence,
            "reasons": reasons,
            "indicators_snapshot": snapshot,
            "stop_loss": None,
            "take_profit": None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/signal/current")
async def signal_current(symbol: str,
                         timeframe: Literal['weekly', 'daily', 'intraday'] = 'weekly',
                         source: Literal['yahoo', 'msn'] = 'yahoo',
                         llm_key: Optional[str] = Header(default=None, alias='X-LLM-KEY'),
                         llm_provider: Optional[str] = Header(default='openai', alias='X-LLM-PROVIDER'),
                         llm_model: Optional[str] = Header(default=None, alias='X-LLM-MODEL')):
    req = AnalyzeRequest(symbol=symbol, timeframe=timeframe, market='IN', source=source)
    return await analyze(req, llm_key=llm_key, llm_provider=llm_provider, llm_model=llm_model)


# ---------------- Portfolio ----------------
@api.get("/portfolio/watchlist")
async def get_watchlist(request: Request):
    user = await _require_user(request)
    symbols = user.get('watchlist') or ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    return {"symbols": symbols}


@api.put("/portfolio/watchlist")
async def put_watchlist(request: Request, body: WatchlistUpdate):
    user = await _require_user(request)
    await db.users.update_one({"id": user['id']}, {"$set": {"watchlist": body.symbols}})
    return {"ok": True, "symbols": body.symbols}


# ---------------- Alerts ----------------
@api.get("/alerts/telegram/config")
async def get_tg_cfg(request: Request):
    user = await _require_user(request)
    cfg = user.get('alert_settings') or {"enabled": False, "buy_threshold": 80, "sell_threshold": 60, "frequency_min": 60, "quiet_start_hour": 22, "quiet_end_hour": 7, "timezone": 'Asia/Kolkata'}
    chat_id = cfg.get('chat_id')
    return {"chat_id": chat_id, **cfg}


@api.post("/alerts/telegram/config")
async def post_tg_cfg(request: Request, cfg: TelegramConfig):
    user = await _require_user(request)
    await db.users.update_one({"id": user['id']}, {"$set": {"alert_settings": cfg.model_dump()}})
    return {"ok": True, **cfg.model_dump()}


@api.post("/alerts/telegram/test")
async def post_tg_test(request: Request, chat_id: Optional[str] = None):
    _ = await _require_user(request)
    # For now, we don't actually send a Telegram message; this is a dry-run endpoint
    return {"ok": True}


@api.get("/alerts/thresholds")
async def get_thresholds(request: Request):
    user = await _require_user(request)
    items = user.get('symbol_thresholds') or {}
    return {"items": items}


@api.post("/alerts/thresholds")
async def post_thresholds(request: Request, payload: SymbolThresholdsPayload):
    user = await _require_user(request)
    mapping: Dict[str, Dict[str, int]] = {}
    for it in payload.items:
        mapping[it.symbol] = {"buy_threshold": it.buy_threshold, "sell_threshold": it.sell_threshold}
    await db.users.update_one({"id": user['id']}, {"$set": {"symbol_thresholds": mapping}})
    return {"ok": True, "items": mapping}


# ---------------- Strategy ----------------
async def score_symbol(sym: str, timeframe: str, source: str, momentum: bool, value: bool) -> Dict[str, Any]:
    try:
        df = await fetch_ohlcv(sym, timeframe, source)
        ind = compute_indicators(df)
        score = ind.get('baseline_confidence', 60.0)
        snap = ind['snapshot']
        if momentum:
            if (snap.get('macd_hist') or 0) > 0:
                score += 5
            if snap.get('sma_50') and snap.get('sma_200') and snap['sma_50'] > snap['sma_200']:
                score += 5
        if value:
            if snap.get('sma_50') and snap.get('price') and snap['price'] < snap['sma_50']:
                score += 5
        score = max(0.0, min(100.0, score))
        return {"ok": True, "ind": ind, "score": score}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def proportional_take(total: int, pct: int, min_each: int = 1) -> int:
    return max(min_each, round((pct / 100.0) * total))


@api.post("/strategy/suggest")
async def strategy_suggest(req: StrategyRequest,
                           llm_key: Optional[str] = Header(default=None, alias='X-LLM-KEY'),
                           llm_provider: Optional[str] = Header(default='openai', alias='X-LLM-PROVIDER'),
                           llm_model: Optional[str] = Header(default=None, alias='X-LLM-MODEL')):
    # Build universe with optional sectors
    universe = await build_universe(req.filters, limit=80)
    results: List[Dict[str, Any]] = []
    horizon = req.filters.horizon if req.filters.horizon in ['daily', 'weekly'] else 'weekly'

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
                "asset_class": 'stocks',
            })

    # RSI filter
    if req.filters.rsi_min is not None or req.filters.rsi_max is not None:
        min_r = req.filters.rsi_min if req.filters.rsi_min is not None else -999
        max_r = req.filters.rsi_max if req.filters.rsi_max is not None else 999

        def ok_rsi(x):
            rsi = x['ind']['snapshot'].get('rsi_14')
            return (rsi is None) or (min_r <= rsi <= max_r)

        results = [x for x in results if ok_rsi(x)]

    # Caps allocation (large/mid/small)
    total_take = max(1, min(req.top_n, 10))
    caps = req.filters.caps_allocation
    buckets = {'largecap': [], 'midcap': [], 'smallcap': []}
    for x in results:
        buckets[x.get('cap', 'largecap')].append(x)
    for k in buckets:
        buckets[k].sort(key=lambda z: z['score'], reverse=True)

    take_large = proportional_take(total_take, caps.get('largecap', 60))
    take_mid = proportional_take(total_take, caps.get('midcap', 25))
    take_small = proportional_take(total_take, caps.get('smallcap', 15))

    selected = buckets['largecap'][:take_large] + buckets['midcap'][:take_mid] + buckets['smallcap'][:take_small]

    if len(selected) > total_take:
        selected.sort(key=lambda z: z['score'], reverse=True)
        selected = selected[:total_take]
    if len(selected) < total_take:
        remaining = [x for x in sorted(results, key=lambda z: z['score'], reverse=True) if x not in selected]
        selected += remaining[: total_take - len(selected)]

    # Build picks
    picks: List[StrategyPick] = []
    for x in selected:
        ind = x['ind']
        action = ind.get('baseline_signal', 'hold')
        reasons = ind.get('baseline_reasons', [])
        picks.append(StrategyPick(symbol=x['symbol'], name=x.get('name'), asset_class='stocks', sector=x.get('sector'), cap=x.get('cap'), score=float(x['score']), action=action, reasons=reasons))

    used_ai = False
    if llm_key and llm_provider == 'openai' and AsyncOpenAI:
        try:
            client_oa = AsyncOpenAI(api_key=llm_key)
            summary = [{"symbol": p.symbol, "score": p.score, "sector": p.sector, "cap": p.cap} for p in picks]
            prompt = f"User filters: {req.filters.model_dump()}\nUser prompt: {req.prompt or ''}\nCandidates: {summary}\nReturn concise reasons per symbol as JSON array of objects {{symbol, short_reason}}."
            resp = await client_oa.responses.create(model=llm_model or os.environ.get('OPENAI_MODEL', 'gpt-5-mini'), input=[{"role": "user", "content": prompt}], max_output_tokens=300)
            txt = getattr(resp, 'output_text', None) or ""
            if txt:
                data = json.loads(txt)
                for p in picks:
                    found = next((d for d in data if d.get('symbol') == p.symbol), None)
                    if found and found.get('short_reason'):
                        p.reasons = [found['short_reason']]
                used_ai = True
        except Exception as e:
            logger.info("Strategy AI refinement skipped: %s", e)

    return {"picks": [p.model_dump() for p in picks], "used_ai": used_ai, "note": "India stocks-only, sector-aware, caps allocation"}


# ---------------- Health ----------------
@api.get("/")
async def root():
    return {"message": "Trading AI backend is up"}


app.include_router(api)
