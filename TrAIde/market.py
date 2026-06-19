"""Live OHLC candle data for the trAIde dashboard's open-position charts.

Read-only, public market data only. Candles are fetched server-side so the browser never
needs a cross-origin call (KuCoin's public API does not send permissive CORS headers) and we
stay inside the same privacy envelope as the rest of the page — price action only, never any
account, size, or identity data.

KuCoin is the primary source because the bot trades there, so the candles line up with the
published entry/TP/SL prices; Binance is a fallback for the rare symbol KuCoin spot doesn't
list. Results are cached briefly in-process so refreshes and multiple viewers don't hammer the
exchanges. Every failure degrades to an empty candle list rather than raising.
"""

import time

import requests

# display interval -> (kucoin `type`, binance `interval`)
_INTERVALS = {
    "15min": ("15min", "15m"),
    "1hour": ("1hour", "1h"),
    "4hour": ("4hour", "4h"),
}
_DEFAULT_INTERVAL = "1hour"
_MAX_CANDLES = 160
_CACHE_TTL = 45      # seconds — matches the dashboard's live refresh cadence
_HTTP_TIMEOUT = 6    # seconds

_cache: dict = {}    # (symbol, interval) -> (expires_ts, candles)


def _valid_symbol(symbol: str) -> bool:
    """Accept only BASE-QUOTE of uppercase alphanumerics, so the proxied outbound URL can never
    be steered to an arbitrary host/path (SSRF guard)."""
    if not symbol or symbol.count("-") != 1:
        return False
    base, quote = symbol.split("-")
    return (
        base.isalnum() and quote.isalnum()
        and 1 <= len(base) <= 12 and 2 <= len(quote) <= 6
    )


def _from_kucoin(symbol: str, ktype: str) -> list:
    """KuCoin spot candles. Row order is [time(s), open, close, high, low, vol, turnover],
    newest first — note open/CLOSE/high/low, not the usual OHLC."""
    r = requests.get(
        "https://api.kucoin.com/api/v1/market/candles",
        params={"type": ktype, "symbol": symbol},
        timeout=_HTTP_TIMEOUT,
    )
    r.raise_for_status()
    rows = (r.json() or {}).get("data") or []
    out = []
    for row in rows:
        try:
            out.append({
                "t": int(row[0]),
                "o": float(row[1]),
                "c": float(row[2]),
                "h": float(row[3]),
                "l": float(row[4]),
            })
        except (TypeError, ValueError, IndexError):
            continue
    out.sort(key=lambda c: c["t"])  # ascending (oldest -> newest)
    return out


def _from_binance(symbol: str, biv: str) -> list:
    """Binance spot klines. Row order is [openTime(ms), open, high, low, close, ...], ascending."""
    r = requests.get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": symbol.replace("-", ""), "interval": biv, "limit": _MAX_CANDLES},
        timeout=_HTTP_TIMEOUT,
    )
    r.raise_for_status()
    out = []
    for row in (r.json() or []):
        try:
            out.append({
                "t": int(row[0]) // 1000,  # ms -> s
                "o": float(row[1]),
                "h": float(row[2]),
                "l": float(row[3]),
                "c": float(row[4]),
            })
        except (TypeError, ValueError, IndexError):
            continue
    out.sort(key=lambda c: c["t"])
    return out


def get_candles(symbol: str, interval: str = _DEFAULT_INTERVAL) -> dict:
    """Recent OHLC candles for a display symbol (e.g. ``BTC-USDT``). Always returns
    ``{"symbol", "interval", "candles"}`` with ``candles`` ascending by time (possibly empty)."""
    symbol = (symbol or "").upper().strip()
    interval = interval if interval in _INTERVALS else _DEFAULT_INTERVAL
    if not _valid_symbol(symbol):
        return {"symbol": symbol, "interval": interval, "candles": []}

    now = time.time()
    ck = (symbol, interval)
    hit = _cache.get(ck)
    if hit and hit[0] > now:
        return {"symbol": symbol, "interval": interval, "candles": hit[1]}

    ktype, biv = _INTERVALS[interval]
    candles: list = []
    for fetch in (lambda: _from_kucoin(symbol, ktype), lambda: _from_binance(symbol, biv)):
        try:
            candles = fetch()
            if candles:
                break
        except Exception:
            candles = []
    candles = candles[-_MAX_CANDLES:]
    # Cache even an empty result briefly so a delisted/unknown symbol isn't retried every poll.
    _cache[ck] = (now + _CACHE_TTL, candles)
    return {"symbol": symbol, "interval": interval, "candles": candles}
