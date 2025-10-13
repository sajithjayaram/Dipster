import logging
import json
from pathlib import Path
from typing import Dict, Any
import httpx
from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the root directory
ROOT_DIR = Path(__file__).parent

# Create FastAPI app
app = FastAPI(title="Stock API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create API router
api_router = APIRouter(prefix="/api")

@api_router.get("/search")
async def search_symbols(q: str = Query(..., min_length=2), region: str = Query('IN')):
    """Yahoo search with offline India fallback."""
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {"q": q, "quotesCount": 10, "newsCount": 0, "listsCount": 0, "enableFuzzyQuery": True, "lang": "en-US", "region": region}
    headers = {"User-Agent": "Mozilla/5.0"}

    async def offline_in_fallback(query: str) -> Dict[str, Any]:
        try:
            import json
            path = ROOT_DIR / 'data' / 'india_tickers.json'
            with open(path, 'r') as f:
                items = json.load(f)
            Q = query.lower()
            filtered = [it for it in items if Q in it['symbol'].lower() or Q in it['name'].lower()]
            return {"results": filtered[:10]}
        except Exception as e:
            logger.exception("Offline fallback failed: %s", e)
            return {"results": []}

    # Prefer online for non-IN or when available
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
                    # fall back if nothing found
                    return await offline_in_fallback(q)
                return {"results": list(dedup.values())[:10]}
            # non-200
            if region.upper() == 'IN':
                return await offline_in_fallback(q)
            raise HTTPException(status_code=502, detail="Search service unavailable")
    except Exception:
        if region.upper() == 'IN':
            return await offline_in_fallback(q)
        raise HTTPException(status_code=502, detail="Search service unavailable")

# Include the API router
app.include_router(api_router)

@app.get("/")
async def root():
    return {"message": "Stock API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
