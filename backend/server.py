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

# Auth deps
from passlib.context import CryptContext
from jose import jwt, JWTError
from starlette.responses import RedirectResponse, JSONResponse

# Optional providers
try:
    import anthropic
except Exception:
    anthropic = None
try:
    import google.generativeai as genai
except Exception:
    genai = None

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'test_database')]

# App & Router
app = FastAPI(title="AI Trading Agent API", version="0.5.1")
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
JWT_EXPIRE_MIN = int(os.environ.get("JWT_EXPIRE_MIN", "10080"))  # 7 days

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# ---------------- Models ----------------
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class AnalyzeRequest(BaseModel):
    symbol: str
    timeframe: Literal['weekly', 'daily', 'intraday'] = 'weekly'
    market: Literal['IN', 'US', 'OTHER'] = 'IN'

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

# Auth & Profile models
class SignupRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class Profile(BaseModel):
    provider: Literal['openai', 'anthropic', 'gemini'] = 'openai'
    model: Optional[str] = Field(default=os.environ.get('OPENAI_MODEL', 'gpt-5-mini'))

class UserOut(BaseModel):
    id: str
    email: EmailStr
    profile: Profile

# Portfolio models
class WatchlistUpdate(BaseModel):
    symbols: List[str]

class TelegramConfig(BaseModel):
    chat_id: Optional[str] = None
    enabled: bool = True
    buy_threshold: int = 80
    sell_threshold: int = 60

# ---------------- Helpers ----------------

def hash_password(pw: str) -> str:
    return pwd_context.hash(pw)

def verify_password(pw: str, hashed: str) -> bool:
    return pwd_context.verify(pw, hashed)

def create_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MIN)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

async def get_current_user(request: Request) -> dict:
    auth = request.headers.get('Authorization')
    if not auth or not auth.lower().startswith('bearer '):
        raise HTTPException(status_code=401, detail="missing_or_invalid_token")
    token = auth.split(' ', 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        uid = payload.get('sub')
        if not uid:
            raise HTTPException(status_code=401, detail="invalid_token")
        user = await db.users.find_one({"id": uid}, {"_id": 0})
        if not user:
            raise HTTPException(status_code=401, detail="user_not_found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="invalid_token")

# ---------------- Market Data & Indicators ----------------
async def fetch_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    if timeframe == 'weekly':
        interval = '1wk'; period = '5y'
    elif timeframe == 'daily':
        interval = '1d'; period = '2y'
    else:
        interval = '60m'; period = '60d'
    try:
        data = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
        if data is None or data.empty:
            raise ValueError("No data returned from Yahoo Finance")
        data = data.dropna().rename(columns=str.title)
        return data
    except Exception as e:
        logger.exception("Failed to fetch data for %s", symbol)
        raise HTTPException(status_code=400, detail=f"Failed to fetch market data: {e}")


def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors='coerce')

    sma_50 = close.rolling(window=50).mean()
    sma_200 = close.rolling(window=200).mean()

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

    last = df.index[-1]
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

    heur_action = 'hold'
    reasons = []
    score = 50.0
    if snapshot['rsi_14'] is not None:
        if snapshot['rsi_14'] < 30:
            reasons.append("RSI indicates oversold (<30)")
            heur_action = 'buy'; score += 15
        elif snapshot['rsi_14'] > 70:
            reasons.append("RSI indicates overbought (>70)")
            heur_action = 'sell'; score -= 15
    if snapshot['sma_50'] and snapshot['sma_200']:
        if snapshot['sma_50'] > snapshot['sma_200']:
            reasons.append("SMA50 above SMA200 (uptrend)")
            if heur_action == 'hold': heur_action = 'buy'
            score += 20
        elif snapshot['sma_50'] < snapshot['sma_200']:
            reasons.append("SMA50 below SMA200 (downtrend)")
            if heur_action == 'hold': heur_action = 'sell'
            score -= 20
    if snapshot['macd_hist'] is not None:
        if snapshot['macd_hist'] > 0:
            reasons.append("MACD histogram positive"); score += 10
        else:
            reasons.append("MACD histogram negative"); score -= 10

    score = max(0.0, min(100.0, score))

    return {'snapshot': snapshot, 'baseline_signal': heur_action, 'baseline_reasons': reasons, 'baseline_confidence': score}

# ---------------- LLM Calls ----------------
async def call_openai(symbol: str, timeframe: str, indicators: Dict[str, Any], api_key: str, model: str) -> AIRecommendation:
    if AsyncOpenAI is None:
        raise HTTPException(status_code=500, detail="OpenAI SDK not available on server")
    client_oa = AsyncOpenAI(api_key=api_key)
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
    try:
        resp = await asyncio.wait_for(
            client_oa.responses.parse(
                model=model or os.environ.get('OPENAI_MODEL', 'gpt-5-mini'),
                input=[{"role": "system", "content": system_instruction}, {"role": "user", "content": user_content}],
                text_format=AITightRecommendation,
                reasoning={"effort": os.environ.get('OPENAI_REASONING', 'low')},
                max_output_tokens=int(os.environ.get('OPENAI_MAX_TOKENS', '350')),
            ),
            timeout=float(os.environ.get('OPENAI_TIMEOUT_SECONDS', '30')),
        )
        tight = resp.output_parsed
        snap = indicators['snapshot']
        conf = float(tight.confidence)
        if conf <= 1.0: conf = round(conf * 100.0, 2)
        conf = max(0.0, min(100.0, conf))
        return AIRecommendation(symbol=tight.symbol, timeframe=tight.timeframe, action=tight.action,
                                confidence=conf, reasons=tight.reasons, indicators_snapshot=snap,
                                stop_loss=tight.stop_loss, take_profit=tight.take_profit)
    except Exception as e:
        logger.exception("OpenAI parse failed, fallback: %s", e)
        snap = indicators['snapshot']
        return AIRecommendation(symbol=symbol, timeframe=timeframe, action=indicators['baseline_signal'], confidence=indicators.get('baseline_confidence', 60.0),
                                reasons=["AI unavailable. Using baseline technical heuristics."] + indicators['baseline_reasons'],
                                indicators_snapshot=snap,
                                stop_loss=round(snap['price'] * (0.95 if indicators['baseline_signal'] == 'buy' else 1.05), 2) if snap.get('price') else None,
                                take_profit=round(snap['price'] * (1.08 if indicators['baseline_signal'] == 'buy' else 0.92), 2) if snap.get('price') else None)

async def call_anthropic(symbol: str, timeframe: str, indicators: Dict[str, Any], api_key: str, model: str) -> AIRecommendation:
    if anthropic is None:
        raise HTTPException(status_code=500, detail="Anthropic SDK not available on server")
    client_an = anthropic.AsyncAnthropic(api_key=api_key)
    system_instruction = "You are a cautious trading analyst. Output JSON strictly to the provided schema."
    prompt = (
        f"Symbol: {symbol}\nTimeframe: {timeframe}\n"
        f"Indicators: {indicators['snapshot']}\n"
        f"Baseline: {indicators['baseline_signal']} | reasons: {indicators['baseline_reasons']}\n"
        "Return only JSON with fields: symbol, timeframe, action(one of buy/sell/hold), confidence(0-100), reasons(list[string]), stop_loss(number|null), take_profit(number|null)."
    )
    try:
        msg = await client_an.messages.create(
            model=model or "claude-3-5-sonnet-20240620",
            max_tokens=400,
            system=system_instruction,
            messages=[{"role": "user", "content": prompt}],
        )
        text_parts = [c.text for c in msg.content if getattr(c, 'type', '') == 'text']
        text = "".join(text_parts) if text_parts else ""
        try:
            data = json.loads(text)
            snap = indicators['snapshot']
            return AIRecommendation(symbol=data.get('symbol', symbol), timeframe=data.get('timeframe', timeframe),
                                    action=data.get('action', indicators['baseline_signal']),
                                    confidence=float(data.get('confidence', 60.0)), reasons=data.get('reasons', []),
                                    indicators_snapshot=snap, stop_loss=data.get('stop_loss'), take_profit=data.get('take_profit'))
        except Exception:
            raise RuntimeError("anthropic_json_parse_error")
    except Exception as e:
        logger.exception("Anthropic failed, fallback: %s", e)
        snap = indicators['snapshot']
        return AIRecommendation(symbol=symbol, timeframe=timeframe, action=indicators['baseline_signal'], confidence=indicators.get('baseline_confidence', 60.0),
                                reasons=["AI unavailable. Using baseline technical heuristics."] + indicators['baseline_reasons'],
                                indicators_snapshot=snap,
                                stop_loss=round(snap['price'] * (0.95 if indicators['baseline_signal'] == 'buy' else 1.05), 2) if snap.get('price') else None,
                                take_profit=round(snap['price'] * (1.08 if indicators['baseline_signal'] == 'buy' else 0.92), 2) if snap.get('price') else None)

async def call_gemini(symbol: str, timeframe: str, indicators: Dict[str, Any], api_key: str, model: str) -> AIRecommendation:
    if genai is None:
        raise HTTPException(status_code=500, detail="Gemini SDK not available on server")
    genai.configure(api_key=api_key)
    mdl = model or "models/gemini-2.5-pro"
    sys_inst = "You are a cautious trading analyst. Return strict JSON only."
    prompt = (
        f"{sys_inst}\n"
        f"Symbol: {symbol}\nTimeframe: {timeframe}\nIndicators: {indicators['snapshot']}\nBaseline: {indicators['baseline_signal']} | {indicators['baseline_reasons']}\n"
        "JSON keys: symbol, timeframe, action (buy|sell|hold), confidence (0-100), reasons (array of strings), stop_loss, take_profit."
    )
    try:
        model_obj = genai.GenerativeModel(mdl)
        resp = await model_obj.generate_content_async(prompt)
        txt = resp.text or ""
        try:
            data = json.loads(txt)
            snap = indicators['snapshot']
            return AIRecommendation(symbol=data.get('symbol', symbol), timeframe=data.get('timeframe', timeframe),
                                    action=data.get('action', indicators['baseline_signal']),
                                    confidence=float(data.get('confidence', 60.0)), reasons=data.get('reasons', []),
                                    indicators_snapshot=snap, stop_loss=data.get('stop_loss'), take_profit=data.get('take_profit'))
        except Exception:
            raise RuntimeError("gemini_json_parse_error")
    except Exception as e:
        logger.exception("Gemini failed, fallback: %s", e)
        snap = indicators['snapshot']
        return AIRecommendation(symbol=symbol, timeframe=timeframe, action=indicators['baseline_signal'], confidence=indicators.get('baseline_confidence', 60.0),
                                reasons=["AI unavailable. Using baseline technical heuristics."] + indicators['baseline_reasons'],
                                indicators_snapshot=snap,
                                stop_loss=round(snap['price'] * (0.95 if indicators['baseline_signal'] == 'buy' else 1.05), 2) if snap.get('price') else None,
                                take_profit=round(snap['price'] * (1.08 if indicators['baseline_signal'] == 'buy' else 0.92), 2) if snap.get('price') else None)

async def call_llm(provider: str, model: Optional[str], api_key: str, symbol: str, timeframe: str, indicators: Dict[str, Any]) -> AIRecommendation:
    provider = (provider or 'openai').lower()
    if not api_key:
        raise HTTPException(status_code=401, detail="missing_user_api_key")
    if provider == 'openai':
        return await call_openai(symbol, timeframe, indicators, api_key, model or os.environ.get('OPENAI_MODEL', 'gpt-5-mini'))
    if provider == 'anthropic':
        return await call_anthropic(symbol, timeframe, indicators, api_key, model or 'claude-3-5-sonnet-20240620')
    if provider == 'gemini':
        return await call_gemini(symbol, timeframe, indicators, api_key, model or 'models/gemini-2.5-pro')
    raise HTTPException(status_code=400, detail="unsupported_provider")

# ---------------- Routes ----------------
@api_router.get("/")
async def root():
    return {"message": "Trading AI backend is up"}

# Auth (email/password)
@api_router.post("/auth/signup", response_model=TokenResponse)
async def signup(req: SignupRequest):
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="password_too_short")
    existing = await db.users.find_one({"email": req.email})
    if existing:
        raise HTTPException(status_code=400, detail="email_in_use")
    user = {
        "id": str(uuid.uuid4()),
        "email": req.email,
        "password_hash": hash_password(req.password),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": {"provider": "openai", "model": os.environ.get('OPENAI_MODEL', 'gpt-5-mini')},
        "watchlist": ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        "alert_settings": {"enabled": False, "buy_threshold": 80, "sell_threshold": 60},
    }
    await db.users.insert_one(user)
    token = create_token(user['id'], user['email'])
    return TokenResponse(access_token=token)

@api_router.post("/auth/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    user = await db.users.find_one({"email": req.email})
    if not user or not verify_password(req.password, user.get('password_hash', '')):
        raise HTTPException(status_code=401, detail="invalid_credentials")
    token = create_token(user['id'], user['email'])
    return TokenResponse(access_token=token)

@api_router.get("/auth/me", response_model=UserOut)
async def me(current=Depends(get_current_user)):
    return UserOut(id=current['id'], email=current['email'], profile=Profile(**current.get('profile', {})))

# Google OAuth
@api_router.get("/auth/google/login")
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
    from urllib.parse import urlencode
    url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    return RedirectResponse(url)

@api_router.get("/auth/google/callback")
async def google_callback(code: Optional[str] = None):
    if not code:
        return JSONResponse({"error": "missing_code"}, status_code=400)
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET or not GOOGLE_REDIRECT_URI:
        raise HTTPException(status_code=500, detail="google_oauth_not_configured")
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
            return JSONResponse({"error": "token_exchange_failed"}, status_code=400)
        token_json = tr.json()
        access_token = token_json.get("access_token")
        if not access_token:
            return JSONResponse({"error": "no_access_token"}, status_code=400)
        ur = await hx.get("https://www.googleapis.com/oauth2/v3/userinfo", headers={"Authorization": f"Bearer {access_token}"})
        if ur.status_code != 200:
            return JSONResponse({"error": "userinfo_failed"}, status_code=400)
        userinfo = ur.json()
    email = userinfo.get("email")
    if not email:
        return JSONResponse({"error": "email_not_found"}, status_code=400)
    existing = await db.users.find_one({"email": email})
    if not existing:
        user = {
            "id": str(uuid.uuid4()),
            "email": email,
            "password_hash": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "profile": {"provider": "openai", "model": os.environ.get('OPENAI_MODEL', 'gpt-5-mini')},
            "watchlist": ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
            "alert_settings": {"enabled": False, "buy_threshold": 80, "sell_threshold": 60},
        }
        await db.users.insert_one(user)
        uid = user['id']
    else:
        uid = existing['id']
    token = create_token(uid, email)
    return RedirectResponse(url=f"/\u003Ftoken={token}")

# Profile
@api_router.get("/profile", response_model=Profile)
async def get_profile(current=Depends(get_current_user)):
    return Profile(**current.get('profile', {}))

@api_router.put("/profile", response_model=Profile)
async def update_profile(p: Profile, current=Depends(get_current_user)):
    await db.users.update_one({"id": current['id']}, {"$set": {"profile": p.model_dump()}})
    return p

# Portfolio (paper)
@api_router.get("/portfolio/watchlist")
async def get_watchlist(current=Depends(get_current_user)):
    user = await db.users.find_one({"id": current['id']}, {"_id": 0, "watchlist": 1})
    return {"symbols": user.get('watchlist', []) if user else []}

@api_router.put("/portfolio/watchlist")
async def put_watchlist(payload: WatchlistUpdate, current=Depends(get_current_user)):
    await db.users.update_one({"id": current['id']}, {"$set": {"watchlist": payload.symbols}})
    return {"ok": True}

# Analysis history
@api_router.get("/portfolio/history")
async def history(symbol: str, timeframe: str = 'weekly', limit: int = 20, current=Depends(get_current_user)):
    cursor = db.analysis_history.find({"user_id": current['id'], "symbol": symbol, "timeframe": timeframe}, {"_id": 0}).sort("created_at", -1).limit(limit)
    return {"items": await cursor.to_list(length=limit)}

# Telegram alerts config
@api_router.get("/alerts/telegram/config")
async def get_telegram_cfg(current=Depends(get_current_user)):
    u = await db.users.find_one({"id": current['id']}, {"_id": 0, "telegram_chat_id": 1, "alert_settings": 1})
    cfg = u.get('alert_settings', {}) if u else {}
    return {
        "chat_id": u.get('telegram_chat_id') if u else None,
        "enabled": bool(cfg.get('enabled', False)),
        "buy_threshold": int(cfg.get('buy_threshold', 80)),
        "sell_threshold": int(cfg.get('sell_threshold', 60)),
    }

@api_router.post("/alerts/telegram/config")
async def set_telegram(cfg: TelegramConfig, current=Depends(get_current_user)):
    await db.users.update_one({"id": current['id']}, {"$set": {"telegram_chat_id": cfg.chat_id, "alert_settings": {"enabled": cfg.enabled, "buy_threshold": cfg.buy_threshold, "sell_threshold": cfg.sell_threshold}}})
    return {"ok": True}

# Search with offline fallback
@api_router.get("/search")
async def search_symbols(q: str = Query(..., min_length=2), region: str = Query('IN')):
    """Yahoo search with offline India fallback."""
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {"q": q, "quotesCount": 10, "newsCount": 0, "listsCount": 0, "enableFuzzyQuery": True, "lang": "en-US", "region": region}
    headers = {"User-Agent": "Mozilla/5.0"}

    async def offline_in_fallback(query: str) -> Dict[str, Any]:
        try:
            path = ROOT_DIR / 'data' / 'india_tickers.json'
            with open(path, 'r') as f:
                items = json.load(f)
            Q = query.lower()
            filtered = [it for it in items if Q in it['symbol'].lower() or Q in it['name'].lower()]
            return {"results": filtered[:10]}
        except Exception as e:
            logger.exception("Offline fallback failed: %s", e)
            return {"results": []}

    try:
        async with httpx.AsyncClient(timeout=10) as client_hx:
            r = await client_hx.get(url, params=params, headers=headers)
            if r.status_code == 200:
                data = r.json() or {}
                quotes = data.get("quotes", [])
                out = []
                for qd in quotes:
                    sym = qd.get("symbol")
                    nm = qd.get("shortname") or qd.get("longname") or qd.get("name")
                    exch = qd.get("exchDisp") or qd.get("exchange")
                    typ = qd.get("quoteType") or qd.get("typeDisp")
                    if not sym or not nm:
                        continue
                    if region.upper() == 'IN' and (sym.endswith('.NS') or sym.endswith('.BO')):
                        out.append({"symbol": sym, "name": nm, "exchange": exch, "type": typ})
                    elif region.upper() != 'IN':
                        out.append({"symbol": sym, "name": nm, "exchange": exch, "type": typ})
                dedup = {it['symbol']: it for it in out}
                if region.upper() == 'IN' and len(dedup) == 0:
                    return await offline_in_fallback(q)
                return {"results": list(dedup.values())[:10]}
            if region.upper() == 'IN':
                return await offline_in_fallback(q)
            raise HTTPException(status_code=502, detail="Search service unavailable")
    except Exception:
        if region.upper() == 'IN':
            return await offline_in_fallback(q)
        raise HTTPException(status_code=502, detail="Search service unavailable")

# Analysis
@api_router.post("/analyze", response_model=AIRecommendation)
async def analyze(req: AnalyzeRequest,
                  request: Request,
                  llm_key: Optional[str] = Header(default=None, alias='X-LLM-KEY'),
                  llm_provider: Optional[str] = Header(default='openai', alias='X-LLM-PROVIDER'),
                  llm_model: Optional[str] = Header(default=None, alias='X-LLM-MODEL')):
    symbol = req.symbol.strip()
    df = await fetch_ohlcv(symbol, req.timeframe)
    indicators = compute_indicators(df)
    rec = await call_llm(llm_provider, llm_model, llm_key, symbol, req.timeframe, indicators)
    if not rec.indicators_snapshot:
        rec.indicators_snapshot = indicators['snapshot']
    # Save history if JWT present
    auth = request.headers.get('Authorization')
    if auth and auth.lower().startswith('bearer '):
        try:
            payload = jwt.decode(auth.split(' ',1)[1], JWT_SECRET, algorithms=[JWT_ALGO])
            uid = payload.get('sub')
            if uid:
                doc = {
                    "user_id": uid,
                    "symbol": rec.symbol,
                    "timeframe": rec.timeframe,
                    "recommendation": rec.model_dump(),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                await db.analysis_history.insert_one(doc)
        except Exception:
            pass
    return rec

@api_router.get("/signal/current", response_model=AIRecommendation)
async def current_signal(symbol: str = Query(...), timeframe: Literal['weekly','daily','intraday'] = 'weekly',
                         llm_key: Optional[str] = Header(default=None, alias='X-LLM-KEY'),
                         llm_provider: Optional[str] = Header(default='openai', alias='X-LLM-PROVIDER'),
                         llm_model: Optional[str] = Header(default=None, alias='X-LLM-MODEL')):
    df = await fetch_ohlcv(symbol.strip(), timeframe)
    indicators = compute_indicators(df)
    rec = await call_llm(llm_provider, llm_model, llm_key, symbol.strip(), timeframe, indicators)
    if not rec.indicators_snapshot:
        rec.indicators_snapshot = indicators['snapshot']
    return rec

# Include router
app.include_router(api_router)

# ---------------- Alerts background (heuristics-based) ----------------
async def alerts_loop():
    if not TELEGRAM_BOT_TOKEN:
        logger.info("Telegram bot token not configured; alerts disabled")
        return
    send_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    while True:
        try:
            users = db.users.find({"alert_settings.enabled": True, "telegram_chat_id": {"$exists": True}})
            async for u in users:
                chat_id = u.get('telegram_chat_id')
                settings = u.get('alert_settings', {})
                buy_th = int(settings.get('buy_threshold', 80))
                sell_th = int(settings.get('sell_threshold', 60))
                watch = u.get('watchlist', [])[:20]
                for sym in watch:
                    try:
                        df = await fetch_ohlcv(sym, 'weekly')
                        indicators = compute_indicators(df)
                        conf = indicators.get('baseline_confidence', 60)
                        action = indicators.get('baseline_signal', 'hold')
                        trigger = (action == 'buy' and conf >= buy_th) or (action == 'sell' and conf >= sell_th)
                        if not trigger:
                            continue
                        last_map = u.get('last_alerts', {})
                        key = f"{sym}_weekly"
                        last_entry = last_map.get(key)
                        if last_entry and last_entry.get('action') == action and (datetime.now(timezone.utc).timestamp() - last_entry.get('ts', 0)) < 3600:
                            continue
                        text = f"Signal {action.upper()} for {sym} (weekly)\nConfidence: {conf}%\nReason: {'; '.join(indicators.get('baseline_reasons', [])[:3])}"
                        async with httpx.AsyncClient(timeout=10) as hx:
                            await hx.post(send_url, json={"chat_id": chat_id, "text": text})
                        last_map[key] = {"action": action, "ts": datetime.now(timezone.utc).timestamp()}
                        await db.users.update_one({"id": u['id']}, {"$set": {"last_alerts": last_map}})
                    except Exception:
                        continue
        except Exception as e:
            logger.exception("alerts loop error: %s", e)
        await asyncio.sleep(300)

@app.on_event("startup")
async def start_alerts():
    asyncio.create_task(alerts_loop())

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
