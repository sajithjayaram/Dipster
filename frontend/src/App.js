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
import { Textarea } from './components/ui/textarea.jsx';
import { BarChart3, Activity, AlertTriangle, Search, User, LogIn, Wand2 } from 'lucide-react';

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
  const [symbols, setSymbols] = useState(() => { try { const s = localStorage.getItem('symbols'); return s ? JSON.parse(s) : defaultIN; } catch { return defaultIN; } });
  const [timeframe, setTimeframe] = useState(() => localStorage.getItem('timeframe') || 'weekly');
  // Lock to Indian markets only
  const [market, setMarket] = useState('IN');
  const [source, setSource] = useState(() => localStorage.getItem('data_source') || 'yahoo');
  const [live, setLive] = useState(() => localStorage.getItem('live_alerts') === '1');
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
  const [tgFreq, setTgFreq] = useState(60);
  const [tgQuietStart, setTgQuietStart] = useState(22);
  const [tgQuietEnd, setTgQuietEnd] = useState(7);
  const [tgTz, setTgTz] = useState('Asia/Kolkata');

  // per-symbol thresholds
  const [thresholdsMap, setThresholdsMap] = useState({});

  // Strategy builder state
  const [risk, setRisk] = useState('medium');
  const [horizon, setHorizon] = useState('weekly');
  const [momentum, setMomentum] = useState(true);
  const [value, setValue] = useState(false);
  const [rsiMin, setRsiMin] = useState('');
  const [rsiMax, setRsiMax] = useState('');
  const [assetStocks, setAssetStocks] = useState(true);
  const [assetETFs, setAssetETFs] = useState(true);
  const [assetComms, setAssetComms] = useState(true);
  const [assetMFs, setAssetMFs] = useState(false);
  const [freePrompt, setFreePrompt] = useState('');
  const [topN, setTopN] = useState(5);
  const [picks, setPicks] = useState([]);
  const [picking, setPicking] = useState(false);

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

  // persist preferences
  useEffect(()=>{ try{ localStorage.setItem('symbols', JSON.stringify(symbols)); }catch{} }, [symbols]);
  useEffect(()=>{ localStorage.setItem('timeframe', timeframe); }, [timeframe]);
  useEffect(()=>{ localStorage.setItem('market', market); }, [market]);
  useEffect(()=>{ localStorage.setItem('data_source', source); }, [source]);
  useEffect(()=>{ localStorage.setItem('live_alerts', live ? '1':'0'); }, [live]);

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
      setTgFreq(res.data.frequency_min ?? 60);
      setTgQuietStart(res.data.quiet_start_hour ?? 22);
      setTgQuietEnd(res.data.quiet_end_hour ?? 7);
      setTgTz(res.data.timezone || 'Asia/Kolkata');
    } catch { /* ignore */ }
  };

  const fetchThresholds = async (tok) => {
    try{
      const res = await axios.get(`${API}/alerts/thresholds`, { headers: { Authorization: `Bearer ${tok}` } });
      setThresholdsMap(res.data?.items || {});
    }catch{}
  };

  useEffect(() => { if (token) { fetchMe(token); fetchWatchlist(token); fetchTelegramCfg(token); fetchThresholds(token); } }, [token]);

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
      await axios.post(`${API}/alerts/telegram/config`, { chat_id: tgChatId || null, enabled: tgEnabled, buy_threshold: tgBuy, sell_threshold: tgSell, frequency_min: tgFreq, quiet_start_hour: tgQuietStart, quiet_end_hour: tgQuietEnd, timezone: tgTz }, { headers: { Authorization: `Bearer ${token}` } });
      const items = symbols.map(sym => ({ symbol: sym, buy_threshold: thresholdsMap[sym]?.buy_threshold ?? tgBuy, sell_threshold: thresholdsMap[sym]?.sell_threshold ?? tgSell }));
      await axios.post(`${API}/alerts/thresholds`, { items }, { headers: { Authorization: `Bearer ${token}` } });
      toast.success('Profile saved');
      setProfileOpen(false);
    } catch (e) {
      toast.error(e?.response?.data?.detail || 'Failed to save profile');
    }
  };

  // ------- SEARCH -------
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [openSearch, setOpenSearch] = useState(false);
  const debounceRef = useRef(null);

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
      const res = await axios.post(`${API}/analyze`, { symbol: s, timeframe, market, source }, {
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
        params: { symbol: s, timeframe, source },
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

  const startPoll = useMemo(() => () => { symbols.forEach(s => pollOne(s)); }, [symbols, timeframe, provider, model, llmKey, token, source]);
  usePolling(live, startPoll, 60000);

  useEffect(() => { symbols.forEach(s => analyze(s)); }, []);

  const openGoogleLogin = () => { window.location.href = `${API}/auth/google/login`; };

  // ------- STRATEGY -------
  const runStrategy = async () => {
    setPicking(true);
    try {
      const asset_classes = [];
      if (assetStocks) asset_classes.push('stocks');
      if (assetETFs) asset_classes.push('etfs');
      if (assetComms) asset_classes.push('commodities');
      if (assetMFs) asset_classes.push('mutual_funds');
      const body = {
        filters: {
          risk_tolerance: risk,
          horizon,
          asset_classes,
          market,
          momentum_preference: momentum,
          value_preference: value,
          rsi_min: rsiMin ? parseInt(rsiMin, 10) : null,
          rsi_max: rsiMax ? parseInt(rsiMax, 10) : null,
          sectors: null,
        },
        prompt: freePrompt || null,
        top_n: topN,
        source,
      };
      const headers = {
        ...(llmKey ? { 'X-LLM-KEY': llmKey, 'X-LLM-PROVIDER': provider, 'X-LLM-MODEL': model } : {}),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      };
      const res = await axios.post(`${API}/strategy/suggest`, body, { headers });
      setPicks(res.data?.picks || []);
      if (res.data?.used_ai) toast.success('Strategy refined by AI'); else toast.success('Strategy ready');
    } catch (e) {
      toast.error(e?.response?.data?.detail || 'Failed to create strategy');
    } finally {
      setPicking(false);
    }
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
                        <div style={{ display:'grid', gridTemplateColumns:'repeat(5,minmax(0,1fr))', gap:10 }}>
                          <div>
                            <Label className="text-sm">Buy ≥</Label>
                            <Input data-testid="tg-buy-input" type="number" min={0} max={100} value={tgBuy} onChange={e=>setTgBuy(parseInt(e.target.value||'80',10))} />
                          </div>
                          <div>
                            <Label className="text-sm">Sell ≥</Label>
                            <Input data-testid="tg-sell-input" type="number" min={0} max={100} value={tgSell} onChange={e=>setTgSell(parseInt(e.target.value||'60',10))} />
                          </div>
                          <div>
                            <Label className="text-sm">Min gap (min)</Label>
                            <Input data-testid="tg-frequency-input" type="number" min={5} max={1440} value={tgFreq} onChange={e=>setTgFreq(parseInt(e.target.value||'60',10))} />
                          </div>
                          <div>
                            <Label className="text-sm">Quiet start</Label>
                            <Input data-testid="tg-quiet-start-input" type="number" min={0} max={23} value={tgQuietStart} onChange={e=>setTgQuietStart(parseInt(e.target.value||'22',10))} />
                          </div>
                          <div>
                            <Label className="text-sm">Quiet end</Label>
                            <Input data-testid="tg-quiet-end-input" type="number" min={0} max={23} value={tgQuietEnd} onChange={e=>setTgQuietEnd(parseInt(e.target.value||'7',10))} />
                          </div>
                        </div>
                        <div>
                          <Label className="text-sm">Timezone</Label>
                          <Input data-testid="tg-timezone-input" value={tgTz} onChange={e=>setTgTz(e.target.value)} />
                        </div>
                      </div>
                      <div className="grid" style={{ gap: 8 }}>
                        <Label className="text-sm">Per-symbol thresholds</Label>
                        <div className="text-xs text-slate-500">Click a value to edit. Unset values fall back to global thresholds.</div>
                        <div className="panel" style={{ padding: 10 }}>
                          {symbols.map(sym => (
                            <div key={sym} style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap: 10, alignItems:'center', marginBottom: 6 }}>
                              <div style={{ fontWeight:600 }}>{sym}</div>
                              <Input data-testid={`sym-${sym}-buy`} type="number" min={0} max={100} value={thresholdsMap[sym]?.buy_threshold ?? ''} placeholder={`Buy ≥ ${tgBuy}`} onChange={e=> setThresholdsMap(prev=> ({ ...prev, [sym]: { ...(prev[sym]||{}), buy_threshold: e.target.value? parseInt(e.target.value,10): undefined } })) } />
                              <Input data-testid={`sym-${sym}-sell`} type="number" min={0} max={100} value={thresholdsMap[sym]?.sell_threshold ?? ''} placeholder={`Sell ≥ ${tgSell}`} onChange={e=> setThresholdsMap(prev=> ({ ...prev, [sym]: { ...(prev[sym]||{}), sell_threshold: e.target.value? parseInt(e.target.value,10): undefined } })) } />
                            </div>
                          ))}
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
              <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 160px 140px 120px 140px', gap: 12 }}>
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
                <div>
                  <Label className="text-sm">Data Source</Label>
                  <Select value={source} onValueChange={setSource}>
                    <SelectTrigger data-testid="source-select"><SelectValue placeholder="Source" /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="yahoo" data-testid="source-yahoo">Yahoo</SelectItem>
                      <SelectItem value="msn" data-testid="source-msn">MSN</SelectItem>
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

            {/* Strategy Builder */}
            <div className="panel" style={{ padding: 16, marginTop: 16 }}>
              <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom: 8 }}>
                <Wand2 size={18} color="#0ea5a4" />
                <div className="font-semibold">Strategy Builder</div>
              </div>
              <div className="grid" style={{ gridTemplateColumns:'repeat(6,minmax(0,1fr))', gap:12 }}>
                <div>
                  <Label className="text-sm">Risk</Label>
                  <Select value={risk} onValueChange={setRisk}>
                    <SelectTrigger data-testid="risk-select"><SelectValue placeholder="Risk" /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="low">Low</SelectItem>
                      <SelectItem value="medium">Medium</SelectItem>
                      <SelectItem value="high">High</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label className="text-sm">Horizon</Label>
                  <Select value={horizon} onValueChange={setHorizon}>
                    <SelectTrigger data-testid="horizon-select"><SelectValue placeholder="Horizon" /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="intraday">Intraday</SelectItem>
                      <SelectItem value="daily">Daily</SelectItem>
                      <SelectItem value="weekly">Weekly</SelectItem>
                      <SelectItem value="longterm">Long term</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label className="text-sm">RSI min</Label>
                  <Input data-testid="rsi-min-input" value={rsiMin} onChange={e=>setRsiMin(e.target.value)} placeholder="e.g. 30" />
                </div>
                <div>
                  <Label className="text-sm">RSI max</Label>
                  <Input data-testid="rsi-max-input" value={rsiMax} onChange={e=>setRsiMax(e.target.value)} placeholder="e.g. 70" />
                </div>
                <div>
                  <Label className="text-sm">Top N</Label>
                  <Input data-testid="topn-input" type="number" min={1} max={10} value={topN} onChange={e=>setTopN(parseInt(e.target.value||'5',10))} />
                </div>
                <div>
                  <Label className="text-sm">Asset classes</Label>
                  <div className="list">
                    <span className="chip" data-testid="asset-stocks" onClick={()=>setAssetStocks(v=>!v)} style={{ background: assetStocks? '#0ea5a4' : undefined, color: assetStocks? '#fff' : undefined }}>Stocks</span>
                    <span className="chip" data-testid="asset-etfs" onClick={()=>setAssetETFs(v=>!v)} style={{ background: assetETFs? '#0ea5a4' : undefined, color: assetETFs? '#fff' : undefined }}>ETFs</span>
                    <span className="chip" data-testid="asset-comms" onClick={()=>setAssetComms(v=>!v)} style={{ background: assetComms? '#0ea5a4' : undefined, color: assetComms? '#fff' : undefined }}>Commodities</span>
                    <span className="chip" data-testid="asset-mfs" onClick={()=>setAssetMFs(v=>!v)} style={{ background: assetMFs? '#0ea5a4' : undefined, color: assetMFs? '#fff' : undefined }}>Mutual Funds</span>
                  </div>
                </div>
              </div>
              <div className="grid" style={{ marginTop: 12 }}>
                <Label className="text-sm">Extra prompt (optional)</Label>
                <Textarea data-testid="strategy-prompt" placeholder="e.g. Prefer largecap IT with positive momentum" value={freePrompt} onChange={e=>setFreePrompt(e.target.value)} />
              </div>
              <div style={{ display:'flex', gap:10, marginTop: 12 }}>
                <Button className="btn-primary" onClick={runStrategy} data-testid="run-strategy-button" disabled={picking}>{picking? 'Building...' : 'Build strategy'}</Button>
              </div>
              {picks.length>0 && (
                <div className="card-grid" style={{ marginTop: 16 }}>
                  {picks.map((p)=> (
                    <div key={p.symbol} className="card card-col-span-6">
                      <Card data-testid={`strategy-pick-${p.symbol}`}>
                        <CardHeader style={{ display:'flex', flexDirection:'row', alignItems:'center', justifyContent:'space-between' }}>
                          <CardTitle className="card-title">{p.symbol} <span className="text-slate-500" style={{ fontWeight:500 }}>• {p.asset_class?.toUpperCase?.() || 'ASSET'}</span></CardTitle>
                          <div style={{ display:'flex', gap:8, alignItems:'center' }}>
                            <span className={`badge ${p.action}`} data-testid={`strategy-action-${p.symbol}`}>{formatAction(p.action)}</span>
                            <Button variant="outline" onClick={()=> addToWatchlist(p.symbol)} data-testid={`strategy-add-${p.symbol}`}>Add</Button>
                            <Button className="btn-primary" onClick={()=> analyze(p.symbol)} data-testid={`strategy-analyze-${p.symbol}`}>Analyze</Button>
                          </div>
                        </CardHeader>
                        <CardContent>
                          <div className="text-sm text-slate-600">Score: {Math.round(p.score)}%</div>
                          {p.reasons?.length>0 && (
                            <ul className="mt-2" style={{ paddingLeft: 18 }}>
                              {p.reasons.map((r, idx)=> (
                                <li key={idx} className="text-sm" data-testid={`strategy-reason-${p.symbol}-${idx}`}>• {r}</li>
                              ))}
                            </ul>
                          )}
                        </CardContent>
                      </Card>
                    </div>
                  ))}
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
                      <Tabs defaultValue="current">
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
                          <div className="text-sm text-slate-500">Login to view history</div>
                        </TabsContent>
                      </Tabs>
                    </CardContent>
                  </Card>
                </div>
              );
            })}
          </div>
        </section>

        <footer className="footer">Built for speed: Strategy Builder + Google login + Profile + Telegram alerts + Watchlist persistence. Set your model/provider and session key to enable analysis.</footer>
      </main>

      <Toaster position="top-right" richColors />
    </div>
  );
}

export default function App() {
  return <Home />;
}
