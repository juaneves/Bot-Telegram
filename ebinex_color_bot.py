import os, time, sys, traceback, atexit
from typing import Dict, Tuple, Optional
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ==========================================

load_dotenv()

# --- Config ---
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").split(",") if s.strip()]
INTERVAL = "1m"  # fixo 1m
EXPIRE_MINUTES = int(os.getenv("EXPIRE_MINUTES", "1"))

# Helpers pra aceitar vírgula no .env
def env_float(key: str, default: float) -> float:
    raw = os.getenv(key, str(default)).strip().replace(',', '.')
    try: return float(raw)
    except ValueError:
        print(f"[WARN] {key} inválido ('{raw}'), usando {default}"); return float(default)
def env_int(key: str, default: int) -> int:
    raw = os.getenv(key, str(default)).strip().replace(',', '.')
    try: return int(float(raw))
    except ValueError:
        print(f"[WARN] {key} inválido ('{raw}'), usando {default}"); return int(default)
def env_bool(key: str, default: str="1") -> bool:
    return os.getenv(key, default).strip() in ("1","true","True","YES","yes")

SLEEP_SECONDS = env_int("SLEEP_SECONDS", 1)
APP_NAME = os.getenv("APP_NAME", "Ebinex 1m Signals")
DEBUG = env_bool("DEBUG", "1")

# Filtros gerais
MIN_BODY_PCT = env_float("MIN_BODY_PCT", 0.00)
MIN_ATR_PCT  = env_float("MIN_ATR_PCT",  0.05)
VOL_MULT     = env_float("VOL_MULT",     1.0)

# Bollinger squeeze
BB_WIDTH_PCT      = env_float("BB_WIDTH_PCT", 0.50)
BB_SQUEEZE_BARS   = env_int("BB_SQUEEZE_BARS", 10)

# Estratégias (toggles)
STRAT_TREND_PULLBACK   = env_bool("STRAT_TREND_PULLBACK", "1")
STRAT_ENGULF_TREND     = env_bool("STRAT_ENGULF_TREND", "1")
STRAT_SQUEEZE_BREAK    = env_bool("STRAT_SQUEEZE_BREAK", "1")
STRAT_REVERSAL_3       = env_bool("STRAT_REVERSAL_3", "1")
STRAT_HALF3_OPPOSITE   = env_bool("STRAT_HALF3_OPPOSITE", "0")

TRIGGER_SECONDS = env_int("TRIGGER_SECONDS", 18)
FIRST_MATCH_WINS = env_bool("FIRST_MATCH_WINS", "1")
SEND_IMMEDIATE = env_bool("SEND_IMMEDIATE", "0")
TIME_SKEW_SECONDS = env_int("TIME_SKEW_SECONDS", 0)

# Cooldown opcional (segundos) — 0 desliga
COOLDOWN_SECONDS = env_int("COOLDOWN_SECONDS", 0)

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": f"{APP_NAME} / python-requests"})

# dedupe: 1 sinal por símbolo por candle
last_signal_key: Dict[str, str] = {}
_last_send_ts: Dict[str, float] = {}

# --- Utils ---
def log(msg: str):
    print(msg, flush=True)

def send_telegram(text: str):
    if not (BOT_TOKEN and CHAT_ID):
        log("[INFO] Telegram não configurado. Mensagem:\n" + text)
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = SESSION.post(url, json={"chat_id": CHAT_ID, "text": text}, timeout=15)
        if r.status_code != 200:
            log(f"[ERRO] Telegram {r.status_code}: {r.text}")
    except Exception as e:
        log(f"[ERRO] Telegram: {e}")

def fetch_klines(symbol: str, limit: int = 600, max_retries: int = 5) -> pd.DataFrame:
    for attempt in range(1, max_retries+1):
        try:
            r = SESSION.get(BINANCE_KLINES, params={"symbol": symbol, "interval": "1m", "limit": limit}, timeout=20)
            if r.status_code == 429:
                wait = min(60, 2*attempt)
                log(f"[RATE] 429 {symbol}. Aguardando {wait}s... ({attempt}/{max_retries})")
                time.sleep(wait); continue
            r.raise_for_status()
            data = r.json()
            cols = ["open_time","open","high","low","close","volume","close_time",
                    "qv","trades","tbb","tbq","ignore"]
            df = pd.DataFrame(data, columns=cols)
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
            return df
        except Exception as e:
            log(f"[ERRO] fetch_klines {symbol}: {e}")
            if DEBUG: traceback.print_exc()
            time.sleep(3)
    raise RuntimeError("Falha ao buscar klines")

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).ewm(alpha=1/n, adjust=False).mean()
    roll_dn = pd.Series(loss, index=series.index).ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([(df["high"]-df["low"]).abs(),
                    (df["high"]-pc).abs(),
                    (df["low"]-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def bbands(s: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = s.rolling(n).mean()
    std = s.rolling(n).std()
    upper = ma + k * std
    lower = ma - k * std
    return lower, ma, upper

def candle_color(o: float, c: float) -> str:
    if c > o: return "green"
    if c < o: return "red"
    return "doji"

# Corpo da vela: escalar + vetorizada
def body_pct(o: float, c: float) -> float:
    base = (o + c) / 2.0
    return (abs(c - o) / base) * 100.0 if base != 0 else 0.0
def body_pct_series(open_s: pd.Series, close_s: pd.Series) -> pd.Series:
    base = (open_s + close_s) / 2.0
    pct = (close_s - open_s).abs() / base.replace(0, np.nan) * 100.0
    return pct.fillna(0.0)

def wick_sizes(o: float, h: float, l: float, c: float) -> Tuple[float,float]:
    top = (h - max(o,c))
    bot = (min(o,c) - l)
    return top, bot

def vol_ok(vol_now: float, vol_ma20: float) -> bool:
    if np.isnan(vol_now) or np.isnan(vol_ma20) or vol_ma20 <= 0:
        return True
    return vol_now >= VOL_MULT * vol_ma20

# --- Estratégias ---
def strat_trend_pullback(df: pd.DataFrame, i: int, min_body: float, min_atr: float) -> Optional[Tuple[str,str]]:
    ema9  = df["ema9"];  ema21 = df["ema21"]
    atrp  = df["atr%"];  bodyp = df["body%"];  volma = df["vol_ma20"]
    # uptrend
    if (ema9[i-1] > ema21[i-1] and
        df.at[i-2,"close"] < df.at[i-2,"open"] and
        df.at[i-1,"close"] > df.at[i-1,"open"] and
        df.at[i-1,"close"] >= ema9[i-1] and
        bodyp[i-1] >= min_body and atrp[i-1] >= min_atr and
        vol_ok(df.at[i-1,"volume"], volma[i-1])):
        return ("BUY", "VERDE")
    # downtrend
    if (ema9[i-1] < ema21[i-1] and
        df.at[i-2,"close"] > df.at[i-2,"open"] and
        df.at[i-1,"close"] < df.at[i-1,"open"] and
        df.at[i-1,"close"] <= ema9[i-1] and
        bodyp[i-1] >= min_body and atrp[i-1] >= min_atr and
        vol_ok(df.at[i-1,"volume"], volma[i-1])):
        return ("SELL", "VERMELHA")
    return None

def strat_engulf_trend(df: pd.DataFrame, i: int, min_body: float) -> Optional[Tuple[str,str]]:
    o1,c1 = df.at[i-1,"open"], df.at[i-1,"close"]
    o2,c2 = df.at[i-2,"open"], df.at[i-2,"close"]
    body1 = body_pct(o1,c1)
    # bullish engulf
    if (c2 < o2 and c1 > o1 and body1 >= min_body and
        (o1 <= c2) and (c1 >= o2) and
        df.at[i-1,"ema21"] > df.at[i-5,"ema21"] and
        df.at[i-1,"rsi"] > 50 and
        vol_ok(df.at[i-1,"volume"], df.at[i-1,"vol_ma20"])):
        return ("BUY","VERDE")
    # bearish engulf
    if (c2 > o2 and c1 < o1 and body1 >= min_body and
        (o1 >= c2) and (c1 <= o2) and
        df.at[i-1,"ema21"] < df.at[i-5,"ema21"] and
        df.at[i-1,"rsi"] < 50 and
        vol_ok(df.at[i-1,"volume"], df.at[i-1,"vol_ma20"])):
        return ("SELL","VERMELHA")
    return None

def strat_squeeze_break(df: pd.DataFrame, i: int) -> Optional[Tuple[str,str]]:
    widths = df["bb_width%"][i-1-BB_SQUEEZE_BARS+1:i].dropna()
    if len(widths) < BB_SQUEEZE_BARS or not (widths < BB_WIDTH_PCT).all():
        return None
    if (df.at[i-1,"close"] > df.at[i-1,"bb_up"] and
        vol_ok(df.at[i-1,"volume"], df.at[i-1,"vol_ma20"])):
        return ("BUY","VERDE")
    if (df.at[i-1,"close"] < df.at[i-1,"bb_dn"] and
        vol_ok(df.at[i-1,"volume"], df.at[i-1,"vol_ma20"])):
        return ("SELL","VERMELHA")
    return None

def strat_reversal_3(df: pd.DataFrame, i: int, min_body: float) -> Optional[Tuple[str,str]]:
    c1 = "green" if df.at[i-1,"close"]>df.at[i-1,"open"] else "red"
    c2 = "green" if df.at[i-2,"close"]>df.at[i-2,"open"] else "red"
    c3 = "green" if df.at[i-3,"close"]>df.at[i-3,"open"] else "red"
    b1 = df.at[i-1,"body%"]; b2 = df.at[i-2,"body%"]; b3 = df.at[i-3,"body%"]
    if min(b1,b2,b3) < min_body:
        return None
    top, bot = wick_sizes(df.at[i-1,"open"], df.at[i-1,"high"], df.at[i-1,"low"], df.at[i-1,"close"])
    body_abs = abs(df.at[i-1,"close"] - df.at[i-1,"open"])
    if body_abs <= 0:
        return None
    if c1==c2==c3=="green" and top >= body_abs:
        return ("SELL","VERMELHA")
    if c1==c2==c3=="red" and bot >= body_abs:
        return ("BUY","VERDE")
    return None

def strat_half3_opposite(df: pd.DataFrame, now_ms: int, open_ms: int, close_ms: int) -> Optional[Tuple[str,str,str]]:
    last_closed = df.iloc[-2]; prev1 = df.iloc[-3]; live = df.iloc[-1]
    def col_body(row): return candle_color(float(row["open"]), float(row["close"])), body_pct(float(row["open"]), float(row["close"]))
    c1,b1 = col_body(last_closed); c2,b2 = col_body(prev1)
    same2 = (c1==c2) and (c1 in ("green","red")) and (b1>=MIN_BODY_PCT and b2>=MIN_BODY_PCT)
    progress = (now_ms - open_ms) / max(1, (close_ms - open_ms))
    live_color,_ = col_body(live)
    if same2 and progress >= 0.5 and live_color == c1:
        opposite = "VERMELHA" if c1 == "green" else "VERDE"
        acao = "VENDA" if opposite=="VERMELHA" else "COMPRA"
        return (acao, opposite, "half3opp")
    return None

# --- Núcleo ---
def analyze_symbol(symbol: str):
    df = fetch_klines(symbol, limit=600)
    if df is None or len(df) < 60: return

    # Indicadores
    df["ema9"]  = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    df["rsi"]   = rsi(df["close"], 14)
    df["atr%"]  = atr(df)/df["close"]*100
    df["body%"] = body_pct_series(df["open"], df["close"])
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    bb_dn, bb_ma, bb_up = bbands(df["close"], 20, 2.0)
    df["bb_dn"] = bb_dn; df["bb_up"] = bb_up
    df["bb_width%"] = (bb_up - bb_dn) / df["close"] * 100

    # Tempo da vela em formação
    live = df.iloc[-1]
    try:
        open_ms  = int(live["open_time"].value // 1_000_000)
        close_ms = int(live["close_time"].value // 1_000_000)
        now_ms   = int(pd.Timestamp.utcnow().value // 1_000_000)
        secs_left = max(0, (close_ms - now_ms) // 1000)
    except Exception:
        open_ms = 0; close_ms = 0; secs_left = 0

    # Ajuste fino broker x Binance
    secs_left = max(0, secs_left - TIME_SKEW_SECONDS)

    # Janela de disparo (ou modo teste)
    if not SEND_IMMEDIATE:
        if not (0 < secs_left <= TRIGGER_SECONDS):
            if DEBUG:
                lc = candle_color(float(live["open"]), float(live["close"]))
                log(f"[DEBUG] {symbol} secs_left={secs_left} | live={lc} | aguardando <= {TRIGGER_SECONDS}s")
            return
    else:
        if DEBUG:
            lc = candle_color(float(live["open"]), float(live["close"]))
            log(f"[DEBUG] {symbol} secs_left={secs_left} | live={lc} | SEND_IMMEDIATE=1")

    i = len(df) - 1
    decisions = []

    # Estratégias (ordem de prioridade)
    if STRAT_TREND_PULLBACK and i >= 22:
        d = strat_trend_pullback(df, i, MIN_BODY_PCT, MIN_ATR_PCT)
        if d:
            decisions.append(("TREND_PULLBACK", d))
            if DEBUG: log(f"[READY] {symbol} TREND_PULLBACK -> {d}")
    if STRAT_ENGULF_TREND and i >= 25:
        d = strat_engulf_trend(df, i, MIN_BODY_PCT)
        if d:
            decisions.append(("ENGULF_TREND", d))
            if DEBUG: log(f"[READY] {symbol} ENGULF_TREND -> {d}")
    if STRAT_SQUEEZE_BREAK and i >= 40:
        d = strat_squeeze_break(df, i)
        if d:
            decisions.append(("SQUEEZE_BREAK", d))
            if DEBUG: log(f"[READY] {symbol} SQUEEZE_BREAK -> {d}")
    if STRAT_REVERSAL_3 and i >= 5:
        d = strat_reversal_3(df, i, MIN_BODY_PCT)
        if d:
            decisions.append(("REVERSAL_3", d))
            if DEBUG: log(f"[READY] {symbol} REVERSAL_3 -> {d}")
    if STRAT_HALF3_OPPOSITE:
        d2 = strat_half3_opposite(df, now_ms, open_ms, close_ms)
        if d2:
            acao, cor, _tag = d2
            decisions.append(("HALF3_OPPOSITE", (acao, cor)))
            if DEBUG: log(f"[READY] {symbol} HALF3_OPPOSITE -> {(acao, cor)}")

    if not decisions:
        return

    # --- DEDUPE POR VELA ---
    # 1 chave por símbolo+candle (open_time). Independentemente da estratégia.
    key = f"{symbol}__{open_ms}"
    if last_signal_key.get(symbol) == key:
        if DEBUG: log(f"[DEDUP] {symbol} já enviou nesta vela ({key})")
        return

    # Escolhe a primeira (prioridade pela ordem acima)
    acao_en, cor = decisions[0][1]
    acao = "COMPRA" if acao_en in ("BUY", "COMPRA") and cor=="VERDE" else ("VENDA" if acao_en in ("SELL","VENDA") and cor=="VERMELHA" else ("COMPRA" if cor=="VERDE" else "VENDA"))

    # --- COOLDOWN OPCIONAL ---
    if COOLDOWN_SECONDS > 0:
        now = time.time()
        last_ts = _last_send_ts.get(symbol, 0)
        if (now - last_ts) < COOLDOWN_SECONDS:
            if DEBUG: log(f"[COOLDOWN] {symbol} aguardando {COOLDOWN_SECONDS-(now-last_ts):.1f}s")
            return

    # Mensagem
    msg = (
        "💎Ebinex Sinal Vip💎\n"
    f"• Par: {symbol}\n"
    "• Tempo: 1m \n"
    f"• Sinal: {acao}\n"
    f"• Próxima vela: {'VERMELHA' if acao=='VENDA' else 'VERDE'}\n"
    "• Até 2 Martingale"
    )
    send_telegram(msg)

    # Marca dedupe/cooldown
    last_signal_key[symbol] = key
    if COOLDOWN_SECONDS > 0:
        _last_send_ts[symbol] = time.time()

    if DEBUG:
        log(f"[SENT] {symbol} {acao}/{cor} key={key}")

def main():
    if not SYMBOLS:
        log("[ERRO] Configure SYMBOLS no .env"); sys.exit(1)
    log(f"[OK] {APP_NAME} | Symbols: {', '.join(SYMBOLS)} | TF: 1m | DEBUG={DEBUG}")
    while True:
        try:
            for sym in SYMBOLS:
                try:
                    analyze_symbol(sym)
                except Exception as e:
                    log(f"[ERRO] analyze_symbol {sym}: {e}")
                    if DEBUG: traceback.print_exc()
            time.sleep(SLEEP_SECONDS)
        except KeyboardInterrupt:
            log("\n[STOP] Encerrado pelo usuário.")
            break
        except Exception as e:
            log(f"[FATAL] Loop principal: {e}")
            if DEBUG: traceback.print_exc()
            time.sleep(5)

if __name__ == "__main__":
    main()

