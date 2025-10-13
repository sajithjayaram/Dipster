import { useEffect, useMemo, useRef, useState } from 'react';
import './App.css';
import axios from 'axios';
import { Toaster, toast } from './components/ui/sonner.jsx';
import { Button } from './components/ui/button.jsx';
import { Input } from './components/ui/input.jsx';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card.jsx';
import { Label } from './components/ui/label.jsx';
import { Switch } from './components/ui/switch.jsx';
import { Badge } from './components/ui/badge.jsx';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './components/ui/select.jsx';
import { Separator } from './components/ui/separator.jsx';
import { BarChart3, Activity, AlertTriangle } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function usePolling(enabled, cb, intervalMs = 60000) {
  const timer = useRef(null);
  useEffect(() => {
    if (!enabled) { if (timer.current) clearInterval(timer.current); return; }
    cb();
    timer.current = setInterval(cb, intervalMs);
    return () => { if (timer.current) clearInterval(timer.current); };
  }, [enabled, cb, intervalMs]);
}

const defaultIN = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS'];

function formatAction(action) { return action?.toUpperCase(); }

function Home() {
  const [symbols, setSymbols] = useState(defaultIN);
  const [newSymbol, setNewSymbol] = useState('');
  const [timeframe, setTimeframe] = useState('weekly');
  const [market, setMarket] = useState('IN');
  const [live, setLive] = useState(false);
  const [loadingMap, setLoadingMap] = useState({});
  const [recs, setRecs] = useState({});

  const addSymbol = () => {
    const s = newSymbol.trim().toUpperCase();
    if (!s) return; 
    if (symbols.includes(s)) { toast.info('Already on watchlist'); return; }
    setSymbols(prev => [s, ...prev]);
    setNewSymbol('');
    toast.success(`${s} added`);
  };

  const removeSymbol = (s) => {
    setSymbols(prev => prev.filter(x => x !== s));
    setRecs(prev => { const cp = { ...prev }; delete cp[s]; return cp; });
  };

  const analyze = async (s) => {
    setLoadingMap(m => ({ ...m, [s]: true }));
    try {
      const res = await axios.post(`${API}/analyze`, { symbol: s, timeframe, market });
      setRecs(prev => ({ ...prev, [s]: res.data }));
      toast.success(`Analysis ready for ${s}`);
    } catch (e) {
      console.error(e);
      toast.error(`Failed to analyze ${s}`);
    } finally {
      setLoadingMap(m => ({ ...m, [s]: false }));
    }
  };

  const pollOne = async (s) => {
    try {
      const res = await axios.get(`${API}/signal/current`, { params: { symbol: s, timeframe } });
      setRecs(prev => ({ ...prev, [s]: res.data }));
    } catch (e) {
      /* silent */
    }
  };

  const startPoll = useMemo(() => () => {
    symbols.forEach(s => pollOne(s));
  }, [symbols, timeframe]);

  usePolling(live, startPoll, 60000);

  useEffect(() => {
    // Initial fetch once
    symbols.forEach(s => analyze(s));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="app-shell">
      <div className="header">
        <div className="container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}>
          <div className="brand">
            <BarChart3 size={22} color="#0ea5a4" />
            <div className="brand-title" style={{ fontSize: 18 }}>TradeSense AI</div>
            <span className="brand-badge" data-testid="brand-badge">Weekly • India</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <Label htmlFor="live-switch" className="text-sm">Live alerts</Label>
            <Switch id="live-switch" checked={live} onCheckedChange={setLive} data-testid="live-alerts-switch" />
          </div>
        </div>
      </div>

      <main className="container" style={{ flex: 1 }}>
        <section className="hero">
          <div>
            <h1 className="text-4xl md:text-5xl font-bold" data-testid="hero-title">AI buy/sell guidance for Indian markets</h1>
            <p className="mt-3 text-slate-600 max-w-2xl" data-testid="hero-subtitle">
              Track stocks, mutual funds and commodities. Get weekly recommendations with clear reasoning. Data powered by Yahoo Finance.
            </p>
            <div className="panel" style={{ padding: 16, marginTop: 16 }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 160px 140px 120px', gap: 12 }}>
                <div>
                  <Label htmlFor="symbol" className="text-sm">Add symbol (e.g., RELIANCE.NS)</Label>
                  <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
                    <Input data-testid="symbol-input" id="symbol" placeholder="RELIANCE.NS" value={newSymbol} onChange={e => setNewSymbol(e.target.value)} />
                    <Button data-testid="add-symbol-button" onClick={addSymbol} className="btn-primary">Add</Button>
                  </div>
                </div>
                <div>
                  <Label className="text-sm">Timeframe</Label>
                  <Select value={timeframe} onValueChange={setTimeframe}>
                    <SelectTrigger data-testid="timeframe-select"><SelectValue placeholder="Select timeframe" /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="weekly" data-testid="timeframe-weekly">Weekly</SelectItem>
                      <SelectItem value="daily" data-testid="timeframe-daily">Daily</SelectItem>
                      <SelectItem value="intraday" data-testid="timeframe-intraday">Intraday (60m)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label className="text-sm">Market</Label>
                  <Select value={market} onValueChange={setMarket}>
                    <SelectTrigger data-testid="market-select"><SelectValue placeholder="Market" /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="IN" data-testid="market-in">India</SelectItem>
                      <SelectItem value="US" data-testid="market-us">US</SelectItem>
                      <SelectItem value="OTHER" data-testid="market-other">Other</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="list" style={{ marginTop: 12 }}>
                {symbols.map(s => (
                  <span key={s} className="chip" data-testid={`watchlist-chip-${s}`} onClick={() => analyze(s)}>
                    {s}
                  </span>
                ))}
              </div>
            </div>
          </div>
          <div className="panel" style={{ padding: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Activity size={18} color="#0ea5a4" />
              <div className="text-sm text-slate-600">Tip: NSE tickers use .NS suffix. Example: TCS.NS, INFY.NS, RELIANCE.NS</div>
            </div>
            <Separator className="my-4" />
            <div className="text-xs text-slate-500">Data from Yahoo Finance. AI: OpenAI GPT‑5 (server-side). We never hardcode URLs; using environment config.</div>
          </div>
        </section>

        <section style={{ marginTop: 24 }}>
          <div className="card-grid">
            {symbols.map((s) => {
              const rec = recs[s];
              const loading = !!loadingMap[s];
              const action = rec?.action || 'hold';
              return (
                <div key={s} className="card card-col-span-6">
                  <Card data-testid={`analysis-card-${s}`}>
                    <CardHeader style={{ display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
                      <CardTitle className="card-title">{s}</CardTitle>
                      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                        <span className={`badge ${action}`} data-testid={`action-badge-${s}`}>{formatAction(action)}</span>
                        <Button data-testid={`analyze-button-${s}`} className="btn-primary" onClick={() => analyze(s)} disabled={loading}>
                          {loading ? 'Analyzing…' : 'Analyze'}
                        </Button>
                      </div>
                    </CardHeader>
                    <CardContent>
                      {rec ? (
                        <div>
                          <div className="text-sm text-slate-600" data-testid={`confidence-${s}`}>Confidence: {Math.round(rec.confidence)}%</div>
                          <div className="text-sm text-slate-600" data-testid={`price-${s}`}>Price: {rec?.indicators_snapshot?.price ? rec.indicators_snapshot.price.toFixed(2) : '—'}</div>
                          <div className="text-sm text-slate-600" data-testid={`rsi-${s}`}>RSI(14): {rec?.indicators_snapshot?.rsi_14 ? rec.indicators_snapshot.rsi_14.toFixed(2) : '—'}</div>
                          <div className="text-sm text-slate-600" data-testid={`macd-${s}`}>MACD hist: {rec?.indicators_snapshot?.macd_hist ? rec.indicators_snapshot.macd_hist.toFixed(4) : '—'}</div>
                          {rec.stop_loss && <div className="text-sm text-slate-600" data-testid={`stoploss-${s}`}>Stop-loss: {rec.stop_loss}</div>}
                          {rec.take_profit && <div className="text-sm text-slate-600" data-testid={`takeprofit-${s}`}>Take-profit: {rec.take_profit}</div>}
                          <ul className="mt-2" style={{ paddingLeft: 18 }}>
                            {(rec.reasons || []).map((r, idx) => (
                              <li key={idx} className="text-sm" data-testid={`reason-${s}-${idx}`}>• {r}</li>
                            ))}
                          </ul>
                        </div>
                      ) : (
                        <div className="text-sm text-slate-500" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <AlertTriangle size={16} /> No analysis yet.
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </div>
              );
            })}
          </div>
        </section>

        <footer className="footer">Built for speed: MVP with robust backend endpoints under /api. Live polling every 60s when enabled.</footer>
      </main>

      <Toaster position="top-right" richColors />
    </div>
  );
}

export default function App() {
  return <Home />;
}
