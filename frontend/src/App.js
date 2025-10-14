import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import './App.css';
import axios from 'axios';
import { Toaster, toast } from './components/ui/sonner.jsx';
import { Button } from './components/ui/button.jsx';
import { Input } from './components/ui/input.jsx';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card.jsx';
import { Label } from './components/ui/label.jsx';
import { Switch } from './components/ui/switch.jsx';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './components/ui/select.jsx';
import { Separator } from './components/ui/separator.jsx';
import { Command, CommandInput, CommandList, CommandItem, CommandEmpty } from './components/ui/command.jsx';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter, DialogTrigger } from './components/ui/dialog.jsx';
import { Tabs, TabsList, TabsTrigger, TabsContent } from './components/ui/tabs.jsx';
import { BarChart3, Activity, AlertTriangle, Search, User, LogIn } from 'lucide-react';

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
  // core app state
  const [symbols, setSymbols] = useState(defaultIN);
  const [timeframe, setTimeframe] = useState('weekly');
  const [market, setMarket] = useState('IN');
  const [live, setLive] = useState(false);
  const [loadingMap, setLoadingMap] = useState({});
  const [recs, setRecs] = useState({});

  // auth state
  const [token, setToken] = useState(() => localStorage.getItem('token') || '');
  const [email, setEmail] = useState('');
  const [authOpen, setAuthOpen] = useState(false);
  const [isSignup, setIsSignup] = useState(false);
  const [authEmail, setAuthEmail] = useState('');
  const [authPassword, setAuthPassword] = useState('');

  // profile state (provider/model) and per-session key
  const [profileOpen, setProfileOpen] = useState(false);
  const [provider, setProvider] = useState(() => localStorage.getItem('llm_provider') || 'openai');
  const [model, setModel] = useState(() => localStorage.getItem('llm_model') || 'gpt-5-mini');
  const [llmKey, setLlmKey] = useState(() => localStorage.getItem('llm_key') || '');

  // telegram settings
  const [tgEnabled, setTgEnabled] = useState(false);
  const [tgChatId, setTgChatId] = useState('');
  const [tgBuy, setTgBuy] = useState(80);
  const [tgSell, setTgSell] = useState(60);

  // search state
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [openSearch, setOpenSearch] = useState(false);
  const debounceRef = useRef(null);

  // history state
  const [activeTab, setActiveTab] = useState('current');
  const [historyMap, setHistoryMap] = useState({}); // { `${symbol}-${timeframe}`: items[] }

  const isAuthed = !!token;
  const loadedWatchlistOnce = useRef(false);

  const setSession = (t) => {
    if (t) { localStorage.setItem('token', t); setToken(t); }
    else { localStorage.removeItem('token'); setToken(''); }
  };

  const saveSessionLLM = (prov, mdl, key) => {
    if (prov) { localStorage.setItem('llm_provider', prov); setProvider(prov); }
    if (mdl) { localStorage.setItem('llm_model', mdl); setModel(mdl); }
    if (key !== undefined) { localStorage.setItem('llm_key', key); setLlmKey(key); }
  };

  // Capture JWT from Google callback (?token=...)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const t = params.get('token');
    if (t) {
      setSession(t);
      window.history.replaceState(null, '', window.location.pathname);
      toast.success('Logged in with Google');
    }
  }, []);

  // ------- AUTH -------
  const fetchMe = async (tok) => {
    try {
      const res = await axios.get(`${API}/auth/me`, { headers: { Authorization: `Bearer ${tok}` } });
      setEmail(res.data.email);
      setProvider(res.data.profile?.provider || 'openai');
      setModel(res.data.profile?.model || 'gpt-5-mini');
    } catch { /* ignore */ }
  };

  const fetchWatchlist = async (tok) => {
    try {
      const res = await axios.get(`${API}/portfolio/watchlist`, { headers: { Authorization: `Bearer ${tok}` } });
      if (Array.isArray(res.data?.symbols) && res.data.symbols.length) {
        setSymbols(res.data.symbols);
      }
      loadedWatchlistOnce.current = true;
    } catch { loadedWatchlistOnce.current = true; }
  };

  const fetchTelegramCfg = async (tok) => {
    try {
      const res = await axios.get(`${API}/alerts/telegram/config`, { headers: { Authorization: `Bearer ${tok}` } });
      setTgEnabled(!!res.data.enabled);
      setTgChatId(res.data.chat_id || '');
      setTgBuy(res.data.buy_threshold ?? 80);
      setTgSell(res.data.sell_threshold ?? 60);
    } catch { /* ignore */ }
  };

  useEffect(() => { if (token) { fetchMe(token); fetchWatchlist(token); fetchTelegramCfg(token); } }, [token]);

  const handleAuth = async () => {
    if (!authEmail || !authPassword) { toast.error('Enter email and password'); return; }
    try {
      const url = isSignup ? `${API}/auth/signup` : `${API}/auth/login`;
      const res = await axios.post(url, { email: authEmail, password: authPassword });
      setSession(res.data.access_token);
      toast.success(isSignup ? 'Signed up' : 'Logged in');
      setAuthOpen(false);
      setAuthEmail(''); setAuthPassword('');
    } catch (e) {
      toast.error(e?.response?.data?.detail || 'Auth failed');
    }
  };

  const handleLogout = () => {
    setSession('');
    setEmail('');
  };

  // ------- PROFILE -------
  const saveProfile = async () => {
    if (!isAuthed) { toast.error('Login required'); return; }
    try {
      await axios.put(`${API}/profile`, { provider, model }, { headers: { Authorization: `Bearer ${token}` } });
      saveSessionLLM(provider, model, llmKey);
      // save telegram cfg too
      await axios.post(`${API}/alerts/telegram/config`, { chat_id: tgChatId || null, enabled: tgEnabled, buy_threshold: tgBuy, sell_threshold: tgSell }, { headers: { Authorization: `Bearer ${token}` } });
      toast.success('Profile saved');
      setProfileOpen(false);
    } catch (e) {
      toast.error(e?.response?.data?.detail || 'Failed to save profile');
    }
  };

  // ------- SEARCH -------
  const fetchSearch = useCallback((q) => {
    if (!q || q.length < 2) { setResults([]); return; }
    axios.get(`${API}/search`, { params: { q, region: market } })
      .then(res => setResults(res.data?.results || []))
      .catch(() => setResults([]));
  }, [market]);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => fetchSearch(query), 300);
  }, [query, fetchSearch]);

  const addToWatchlist = (s) => {
    const S = (s || '').trim().toUpperCase();
    if (!S) return;
    if (symbols.includes(S)) { toast.info('Already on watchlist'); return; }
    const next = [S, ...symbols];
    setSymbols(next);
    setQuery(''); setResults([]); setOpenSearch(false);
    toast.success(`${S} added`);
    // persist server-side if authed
    if (isAuthed && loadedWatchlistOnce.current) {
      axios.put(`${API}/portfolio/watchlist`, { symbols: next }, { headers: { Authorization: `Bearer ${token}` } }).catch(()=>{});
    }
  };

  const removeFromWatchlist = (s) => {
    const next = symbols.filter(x => x !== s);
    setSymbols(next);
    if (isAuthed && loadedWatchlistOnce.current) {
      axios.put(`${API}/portfolio/watchlist`, { symbols: next }, { headers: { Authorization: `Bearer ${token}` } }).catch(()=>{});
    }
  };

  // ------- ANALYZE -------
  const analyze = async (s) => {
    if (!llmKey) { toast.error('Set your API key in Profile'); return; }
    setLoadingMap(m => ({ ...m, [s]: true }));
    try {
      const res = await axios.post(`${API}/analyze`, { symbol: s, timeframe, market }, {
        headers: {
          'X-LLM-KEY': llmKey,
          'X-LLM-PROVIDER': provider,
          'X-LLM-MODEL': model,
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        }
      });
      setRecs(prev => ({ ...prev, [s]: res.data }));
      toast.success(`Analysis ready for ${s}`);
    } catch (e) {
      toast.error(e?.response?.data?.detail || `Failed to analyze ${s}`);
    } finally {
      setLoadingMap(m => ({ ...m, [s]: false }));
    }
  };

  const pollOne = async (s) => {
    if (!llmKey) return;
    try {
      const res = await axios.get(`${API}/signal/current`, {
        params: { symbol: s, timeframe },
        headers: {
          'X-LLM-KEY': llmKey,
          'X-LLM-PROVIDER': provider,
          'X-LLM-MODEL': model,
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        }
      });
      setRecs(prev => ({ ...prev, [s]: res.data }));
    } catch { /* silent */ }
  };

  const startPoll = useMemo(() => () => { symbols.forEach(s => pollOne(s)); }, [symbols, timeframe, provider, model, llmKey, token]);
  usePolling(live, startPoll, 60000);

  useEffect(() => { symbols.forEach(s => analyze(s)); }, []);

  const openGoogleLogin = () => {
    window.location.href = `${API}/auth/google/login`;
  };

  const fetchHistory = async (s) => {
    if (!isAuthed) return;
    const key = `${s}-${timeframe}`;
    if (historyMap[key]) return; // already loaded
    try {
      const res = await axios.get(`${API}/portfolio/history`, { params: { symbol: s, timeframe, limit: 10 }, headers: { Authorization: `Bearer ${token}` } });
      setHistoryMap(prev => ({ ...prev, [key]: res.data?.items || [] }));
    } catch { /* ignore */ }
  };

  return (
    <div className="app-shell">
      <div className="header">
        <div className="container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}>
          <div className="brand">
            <BarChart3 size={22} color="#0ea5a4" />
            <div className="brand-title" style={{ fontSize: 18 }}>TradeSense AI</div>
            <span className="brand-badge" data-testid="brand-badge">{timeframe === 'weekly' ? 'Weekly' : timeframe === 'daily' ? 'Daily' : 'Intraday'} • {market === 'IN' ? 'India' : market === 'US' ? 'US' : 'Other'}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <Label htmlFor="live-switch" className="text-sm">Live alerts</Label>
            <Switch id="live-switch" checked={live} onCheckedChange={(v)=>{ if(!llmKey && v){ toast.info('Set your API key in Profile'); return;} setLive(v); }} data-testid="live-alerts-switch" />
            {isAuthed ? (
              <div style={{ display:'flex', alignItems:'center', gap:8 }}>
                <span className="text-sm" data-testid="user-email">{email || 'Logged in'}</span>
                <Dialog open={profileOpen} onOpenChange={setProfileOpen}>
                  <DialogTrigger asChild>
                    <Button className="btn-primary" data-testid="open-profile-button"><User size={14} style={{ marginRight:6 }} /> Profile</Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Profile: Model, Key & Alerts</DialogTitle>
                      <DialogDescription>Choose provider/model, set your session API key, and manage Telegram alerts. Keys are never stored on server.</DialogDescription>
                    </DialogHeader>
                    <div className="grid" style={{ gap: 12 }}>
                      <div>
                        <Label className="text-sm">Provider</Label>
                        <Select value={provider} onValueChange={setProvider}>
                          <SelectTrigger data-testid="provider-select"><SelectValue placeholder="Provider" /></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="openai" data-testid="provider-openai">OpenAI</SelectItem>
                            <SelectItem value="anthropic" data-testid="provider-anthropic">Claude</SelectItem>
                            <SelectItem value="gemini" data-testid="provider-gemini">Gemini</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label className="text-sm">Model</Label>
                        <Input data-testid="model-input" placeholder={provider==='openai'?'gpt-5-mini':provider==='anthropic'?'claude-3-5-sonnet-20240620':'models/gemini-2.5-pro'} value={model} onChange={e=>setModel(e.target.value)} />
                      </div>
                      <div>
                        <Label className="text-sm">Session API key</Label>
                        <Input data-testid="session-key-input" type="password" placeholder="Paste your API key" value={llmKey} onChange={e=>setLlmKey(e.target.value)} />
                      </div>
                      <Separator />
                      <div style={{ display:'grid', gap:8 }}>
                        <Label className="text-sm">Telegram Alerts</Label>
                        <div style={{ display:'flex', alignItems:'center', gap:10 }}>
                          <Switch checked={tgEnabled} onCheckedChange={setTgEnabled} data-testid="tg-enabled-switch" />
                          <span className="text-xs text-slate-600">Enable alerts</span>
                        </div>
                        <div>
                          <Label className="text-sm">Chat ID</Label>
                          <Input data-testid="tg-chatid-input" placeholder="123456789" value={tgChatId} onChange={e=>setTgChatId(e.target.value)} />
                        </div>
                        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:10 }}>
                          <div>
                            <Label className="text-sm">Buy threshold</Label>
                            <Input data-testid="tg-buy-input" type="number" min={0} max={100} value={tgBuy} onChange={e=>setTgBuy(parseInt(e.target.value||'80',10))} />
                          </div>
                          <div>
                            <Label className="text-sm">Sell threshold</Label>
                            <Input data-testid="tg-sell-input" type="number" min={0} max={100} value={tgSell} onChange={e=>setTgSell(parseInt(e.target.value||'60',10))} />
                          </div>
                        </div>
                      </div>
                    </div>
                    <DialogFooter>
                      <div style={{ display:'flex', gap:10 }}>
                        <Button onClick={saveProfile} className="btn-primary" data-testid="save-profile-button">Save</Button>
                        <Button variant="outline" onClick={async()=>{
                          if(!token){ toast.error('Login required'); return; }
                          try{
                            await axios.post(`${API}/alerts/telegram/test`, { chat_id: tgChatId || undefined }, { headers: { Authorization: `Bearer ${token}` } });
                            toast.success('Test alert sent');
                          }catch(e){
                            toast.error(e?.response?.data?.detail || 'Failed to send test alert');
                          }
                        }} data-testid="send-test-alert-button">Send test alert</Button>
                      </div>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
                <Button variant="outline" onClick={handleLogout} data-testid="logout-button">Logout</Button>
              </div>
            ) : (
              <div style={{ display:'flex', alignItems:'center', gap:8 }}>
                <Button className="btn-primary" data-testid="google-login-button" onClick={openGoogleLogin}><LogIn size={14} style={{ marginRight:6 }} /> Continue with Google</Button>
                <Dialog open={authOpen} onOpenChange={setAuthOpen}>
                  <DialogTrigger asChild>
                    <Button variant="outline" data-testid="open-login-button">Login / Signup</Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>{isSignup ? 'Create account' : 'Login'}</DialogTitle>
                      <DialogDescription>Email/password auth. No keys are stored.</DialogDescription>
                    </DialogHeader>
                    <div className="grid" style={{ gap: 12 }}>
                      <div>
                        <Label className="text-sm">Email</Label>
                        <Input data-testid="auth-email-input" value={authEmail} onChange={e=>setAuthEmail(e.target.value)} placeholder="you@example.com" />
                      </div>
                      <div>
                        <Label className="text-sm">Password</Label>
                        <Input data-testid="auth-password-input" type="password" value={authPassword} onChange={e=>setAuthPassword(e.target.value)} placeholder="••••••••" />
                      </div>
                      <div style={{ display:'flex', gap:8 }}>
                        <Button onClick={handleAuth} className="btn-primary" data-testid="auth-submit-button">{isSignup ? 'Sign up' : 'Login'}</Button>
                        <Button variant="outline" onClick={()=>setIsSignup(v=>!v)} data-testid="toggle-auth-mode">{isSignup ? 'Switch to Login' : 'Switch to Signup'}</Button>
                      </div>
                    </div>
                  </DialogContent>
                </Dialog>
              </div>
            )}
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
              <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 160px 140px 120px', gap: 12 }}>
                <div>
                  <Label className="text-sm">Add stock or fund</Label>
                  <div style={{ position: 'relative', marginTop: 6 }}>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <Input data-testid="search-input" placeholder="Search (e.g., Reliance, TCS, NIFTY)" value={query} onFocus={()=>setOpenSearch(true)} onChange={e=>{ setQuery(e.target.value); setOpenSearch(true); }} />
                      <Button data-testid="manual-add-button" className="btn-primary" onClick={()=> addToWatchlist(query)}>Add</Button>
                    </div>
                    {openSearch && (results?.length > 0 || query.length >= 2) && (
                      <div className="panel" style={{ position: 'absolute', top: '110%', left: 0, right: 0, zIndex: 40, padding: 0 }} data-testid="search-dropdown">
                        <Command>
                          <div style={{ display:'flex', alignItems:'center', gap:6, padding:'8px 10px' }}>
                            <Search size={14} />
                            <CommandInput placeholder="Type to search" value={query} onValueChange={setQuery} />
                          </div>
                          <CommandList>
                            {results?.length === 0 && <CommandEmpty>No results</CommandEmpty>}
                            {results?.map((r) => (
                              <CommandItem key={r.symbol} onSelect={() => { addToWatchlist(r.symbol); setOpenSearch(false); }} data-testid={`search-result-${r.symbol}`}>
                                <div style={{ display:'flex', justifyContent:'space-between', width:'100%' }}>
                                  <span>{r.name}</span>
                                  <span style={{ color:'#0ea5a4', fontWeight:600 }}>{r.symbol}</span>
                                </div>
                              </CommandItem>
                            ))}
                          </CommandList>
                        </Command>
                      </div>
                    )}
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
              {!llmKey && (
                <div className="text-sm" style={{ color:'#b45309', marginTop: 10 }} data-testid="missing-key-banner">
                  Set your session API key in Profile to enable AI analysis.
                </div>
              )}
            </div>
          </div>
          <div className="panel" style={{ padding: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Activity size={18} color="#0ea5a4" />
              <div className="text-sm text-slate-600">Tip: NSE tickers use .NS suffix. Example: TCS.NS, INFY.NS, RELIANCE.NS</div>
            </div>
            <Separator className="my-4" />
            <div className="text-xs text-slate-500">Providers: OpenAI, Claude, Gemini. Keys are per-session only and never stored server-side.</div>
          </div>
        </section>

        <section style={{ marginTop: 24 }}>
          <div className="card-grid">
            {symbols.map((s) => {
              const rec = recs[s];
              const loading = !!loadingMap[s];
              const action = rec?.action || 'hold';
              const histKey = `${s}-${timeframe}`;
              const historyItems = historyMap[histKey] || [];
              return (
                <div key={s} className="card card-col-span-6">
                  <Card data-testid={`analysis-card-${s}`}>
                    <CardHeader style={{ display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
                      <CardTitle className="card-title">{s}</CardTitle>
                      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                        <span className={`badge ${action}`} data-testid={`action-badge-${s}`}>{formatAction(action)}</span>
                        <Button data-testid={`remove-button-${s}`} variant="outline" onClick={() => removeFromWatchlist(s)}>Remove</Button>
                        <Button data-testid={`analyze-button-${s}`} className="btn-primary" onClick={() => analyze(s)} disabled={loading || !llmKey}>
                          {loading ? 'Analyzing…' : 'Analyze'}
                        </Button>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <Tabs value={activeTab} onValueChange={(v)=>{ setActiveTab(v); if (v==='history') fetchHistory(s); }}>
                        <TabsList>
                          <TabsTrigger value="current" data-testid={`tab-current-${s}`}>Current</TabsTrigger>
                          <TabsTrigger value="history" data-testid={`tab-history-${s}`}>History</TabsTrigger>
                        </TabsList>
                        <TabsContent value="current">
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
                        </TabsContent>
                        <TabsContent value="history">
                          {isAuthed ? (
                            <div className="text-sm text-slate-700" data-testid={`history-list-${s}`}>
                              {historyItems.length === 0 ? (
                                <div className="text-slate-500">No history yet.</div>
                              ) : (
                                <ul style={{ paddingLeft: 18 }}>
                                  {historyItems.map((it, idx) => (
                                    <li key={idx} style={{ marginBottom: 8 }}>
                                      <div style={{ display:'flex', justifyContent:'space-between' }}>
                                        <span>{new Date(it.created_at || it.recommendation?.generated_at || Date.now()).toLocaleString()}</span>
                                        <span style={{ fontWeight:600 }}>{(it.recommendation?.action || '').toUpperCase()} • {Math.round(it.recommendation?.confidence ?? 0)}%</span>
                                      </div>
                                    </li>
                                  ))}
                                </ul>
                              )}
                            </div>
                          ) : (
                            <div className="text-sm text-slate-500">Login to view history</div>
                          )}
                        </TabsContent>
                      </Tabs>
                    </CardContent>
                  </Card>
                </div>
              );
            })}
          </div>
        </section>

        <footer className="footer">Built for speed: Google login + Profile + Telegram alerts + Watchlist persistence (server). Set your model/provider and session key to enable analysis.</footer>
      </main>

      <Toaster position="top-right" richColors />
    </div>
  );
}

export default function App() {
  return <Home />;
}
