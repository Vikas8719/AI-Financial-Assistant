"""
Finnhub service — live prices, news, earnings calendar.
──────────────────────────────────────────────────────────────────
FIX 1: Indian stocks auto .NS suffix + better fallback chain
FIX 2: _get_indian_quote — yfinance data extraction improved
        (was returning None price when info dict has 'currentPrice' key missing)
FIX 3: get_quote now has 3-layer fallback:
        Finnhub → yfinance .NS → web search price
FIX 4: All sync Finnhub calls wrapped in run_in_executor (non-blocking)
FIX 5: company_profile2 call also async now (was blocking event loop)
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

import finnhub

from app.config import settings

logger = logging.getLogger("finbot.finnhub")

# Indian stock name → NSE ticker mapping
INDIAN_STOCKS = {
    "tata steel":    "TATASTEEL.NS",
    "tatasteel":     "TATASTEEL.NS",
    "tcs":           "TCS.NS",
    "infosys":       "INFY.NS",
    "infy":          "INFY.NS",
    "wipro":         "WIPRO.NS",
    "hdfc":          "HDFCBANK.NS",
    "hdfcbank":      "HDFCBANK.NS",
    "reliance":      "RELIANCE.NS",
    "ril":           "RELIANCE.NS",
    "icicibank":     "ICICIBANK.NS",
    "icici":         "ICICIBANK.NS",
    "sbi":           "SBIN.NS",
    "axisbank":      "AXISBANK.NS",
    "bajajfinance":  "BAJFINANCE.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "itc":           "ITC.NS",
    "maruti":        "MARUTI.NS",
    "tatamotors":    "TATAMOTORS.NS",
    "tata motors":   "TATAMOTORS.NS",
    "sunpharma":     "SUNPHARMA.NS",
    "sun pharma":    "SUNPHARMA.NS",
    "hcltech":       "HCLTECH.NS",
    "hcl":           "HCLTECH.NS",
    "ultracemco":    "ULTRACEMCO.NS",
    "asianpaint":    "ASIANPAINT.NS",
    "asian paints":  "ASIANPAINT.NS",
    "kotak":         "KOTAKBANK.NS",
    "kotakbank":     "KOTAKBANK.NS",
    "bhartiairtel":  "BHARTIAIRTEL.NS",
    "airtel":        "BHARTIAIRTEL.NS",
    "adanient":      "ADANIENT.NS",
    "adani":         "ADANIENT.NS",
    "adaniports":    "ADANIPORTS.NS",
    "ntpc":          "NTPC.NS",
    "ongc":          "ONGC.NS",
    "powergrid":     "POWERGRID.NS",
    "titan":         "TITAN.NS",
    "nestleindia":   "NESTLEIND.NS",
    "nestle":        "NESTLEIND.NS",
    "lti":           "LTIM.NS",
    "ltimindtree":   "LTIM.NS",
    "drreddy":       "DRREDDY.NS",
    "dr reddy":      "DRREDDY.NS",
    "cipla":         "CIPLA.NS",
    "divislab":      "DIVISLAB.NS",
    "hindalco":      "HINDALCO.NS",
    "jswsteel":      "JSWSTEEL.NS",
    "jsw steel":     "JSWSTEEL.NS",
    "bajaj auto":    "BAJAJ-AUTO.NS",
    "bajajauto":     "BAJAJ-AUTO.NS",
    "eicher":        "EICHERMOT.NS",
    "hero moto":     "HEROMOTOCO.NS",
    "heromoto":      "HEROMOTOCO.NS",
    "indusindbk":    "INDUSINDBK.NS",
    "indusind":      "INDUSINDBK.NS",
    "zomato":        "ZOMATO.NS",
    "paytm":         "PAYTM.NS",
    "nykaa":         "NYKAA.NS",
    "bpcl":          "BPCL.NS",
    "ioc":           "IOC.NS",
    "coal india":    "COALINDIA.NS",
    "coalindia":     "COALINDIA.NS",
    "upl":           "UPL.NS",
    "grasim":        "GRASIM.NS",
    "shreecem":      "SHREECEM.NS",
    "techm":         "TECHM.NS",
    "tech mahindra": "TECHM.NS",
    "m&m":           "M&M.NS",
    "mahindra":      "M&M.NS",
    "divis":         "DIVISLAB.NS",
    "pidilite":      "PIDILITIND.NS",
    "havells":       "HAVELLS.NS",
    "berger":        "BERGEPAINT.NS",
    "mrf":           "MRF.NS",
    "abbotindia":    "ABBOTINDIA.NS",
}

# US large-cap tickers — skip .NS suffix
US_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "META", "AMZN", "NVDA", "TSLA",
    "NFLX", "AMD", "INTC", "CRM", "ORCL", "IBM", "QCOM", "BABA", "TSM",
    "UBER", "LYFT", "SNAP", "TWTR", "SHOP", "SQ", "PYPL", "V", "MA",
    "JPM", "BAC", "GS", "MS", "WFC", "C", "BRK", "XOM", "CVX",
    "KO", "PEP", "MCD", "SBUX", "DIS", "NFLX", "T", "VZ"
}


def resolve_indian_ticker(symbol: str) -> str:
    """
    Resolve Indian company name → NSE ticker.
    'Tata Steel' → 'TATASTEEL.NS'
    'TATASTEEL'  → 'TATASTEEL.NS' (auto-add .NS if looks Indian)
    """
    key = symbol.strip().lower().replace(" ", "")

    # Direct name match (with spaces)
    lower_sym = symbol.strip().lower()
    if lower_sym in INDIAN_STOCKS:
        return INDIAN_STOCKS[lower_sym]

    # Match without spaces
    for name, ticker in INDIAN_STOCKS.items():
        if key == name.replace(" ", ""):
            return ticker

    # If already has .NS or .BO — return as is
    upper = symbol.upper().strip()
    if ".NS" in upper or ".BO" in upper:
        return upper

    # Known US ticker — return as is
    if upper in US_TICKERS:
        return upper

    return symbol.upper().strip()


class FinnhubService:
    def __init__(self):
        self._client: Optional[finnhub.Client] = None

    @property
    def client(self) -> finnhub.Client:
        """Lazy init — avoids crash if API key missing at startup."""
        if self._client is None:
            self._client = finnhub.Client(api_key=settings.FINNHUB_API_KEY)
        return self._client

    async def get_quote(self, symbol: str) -> dict:
        """
        Get real-time stock quote with 3-layer fallback:
        1. Finnhub (US stocks)
        2. yfinance (Indian .NS/.BO stocks, or Finnhub fails)
        3. Web search price hint
        """
        symbol = resolve_indian_ticker(symbol)

        # Indian stocks → go straight to yfinance (Finnhub doesn't cover NSE well)
        if ".NS" in symbol or ".BO" in symbol:
            result = await self._get_yfinance_quote(symbol)
            if result.get("price"):
                return result
            # Web search fallback
            return await self._web_price_fallback(symbol)

        # US stocks → Finnhub first
        try:
            loop   = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._fetch_quote_sync, symbol),
                timeout=8.0
            )
            if result.get("price"):
                return result
        except asyncio.TimeoutError:
            logger.warning(f"Finnhub timeout for {symbol}")
        except Exception as e:
            logger.warning(f"Finnhub error for {symbol}: {e}")

        # Finnhub failed → try yfinance
        result = await self._get_yfinance_quote(symbol)
        if result.get("price"):
            return result

        # Both failed → web search
        return await self._web_price_fallback(symbol)

    def _fetch_quote_sync(self, symbol: str) -> dict:
        """Blocking Finnhub call — runs in thread pool."""
        try:
            quote = self.client.quote(symbol)
            if not quote:
                return {"error": "Empty response", "symbol": symbol}

            price = quote.get("c")
            if not price:
                return {"error": "No price data", "symbol": symbol}

            # Get company profile (non-fatal if fails)
            profile = {}
            try:
                profile = self.client.company_profile2(symbol=symbol) or {}
            except Exception:
                pass

            return {
                "symbol":     symbol,
                "company":    profile.get("name", symbol),
                "price":      float(price),
                "change":     quote.get("d"),
                "change_pct": quote.get("dp"),
                "high":       quote.get("h"),
                "low":        quote.get("l"),
                "open":       quote.get("o"),
                "prev_close": quote.get("pc"),
                "currency":   profile.get("currency", "USD"),
                "exchange":   profile.get("exchange", ""),
                "industry":   profile.get("finnhubIndustry", ""),
                "market_cap": profile.get("marketCapitalization"),
                "_source":    "finnhub",
                "timestamp":  datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {"error": str(e), "symbol": symbol}

    async def _get_yfinance_quote(self, symbol: str) -> dict:
        """
        yfinance fallback — works for Indian NSE/BSE + US stocks.
        FIX: Improved price extraction — tries multiple keys in order.
        """
        try:
            def _fetch():
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                info   = ticker.info or {}

                # FIX: Try multiple price keys in priority order
                price = (
                    info.get("currentPrice") or
                    info.get("regularMarketPrice") or
                    info.get("lastPrice") or
                    info.get("ask") or
                    info.get("bid") or
                    info.get("previousClose")
                )

                # If still no price, try recent history
                if not price:
                    try:
                        hist = ticker.history(period="1d", timeout=5)
                        if hist is not None and len(hist) > 0:
                            price = float(hist["Close"].iloc[-1])
                    except Exception:
                        pass

                if not price:
                    return {"error": "No price available", "symbol": symbol}

                # Change calculation
                prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
                change = None
                change_pct = None
                if prev_close and price:
                    change = round(float(price) - float(prev_close), 2)
                    change_pct = round((change / float(prev_close)) * 100, 2)
                elif info.get("regularMarketChange"):
                    change = info.get("regularMarketChange")
                    change_pct = info.get("regularMarketChangePercent")

                return {
                    "symbol":     symbol,
                    "company":    info.get("longName") or info.get("shortName", symbol),
                    "price":      float(price),
                    "change":     change,
                    "change_pct": change_pct,
                    "high":       info.get("dayHigh") or info.get("regularMarketDayHigh"),
                    "low":        info.get("dayLow") or info.get("regularMarketDayLow"),
                    "open":       info.get("open") or info.get("regularMarketOpen"),
                    "prev_close": prev_close,
                    "volume":     info.get("volume") or info.get("regularMarketVolume"),
                    "market_cap": info.get("marketCap"),
                    "52w_high":   info.get("fiftyTwoWeekHigh"),
                    "52w_low":    info.get("fiftyTwoWeekLow"),
                    "pe_ratio":   info.get("trailingPE"),
                    "currency":   info.get("currency", "INR" if ".NS" in symbol or ".BO" in symbol else "USD"),
                    "exchange":   info.get("exchange", "NSE" if ".NS" in symbol else ""),
                    "_source":    "yfinance",
                    "timestamp":  datetime.utcnow().isoformat()
                }

            loop   = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _fetch),
                timeout=12.0
            )
            return result

        except asyncio.TimeoutError:
            logger.warning(f"yfinance timeout for {symbol}")
            return {"error": "yfinance timeout", "symbol": symbol}
        except Exception as e:
            logger.warning(f"yfinance error for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}

    async def _web_price_fallback(self, symbol: str) -> dict:
        """
        Web search as absolute last resort for price.
        Returns structured dict with web_price_hint.
        """
        try:
            from app.services.web_search import WebSearchService
            ws = WebSearchService()
            result = await ws.search_stock_price(symbol)
            return {
                "symbol":       symbol,
                "error":        "Live price unavailable — check below",
                "price":        None,
                "web_answer":   result.get("answer", ""),
                "price_hint":   result.get("price_hint", ""),
                "web_results":  result.get("results", [])[:3],
                "_source":      "web_fallback",
                "timestamp":    datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {"error": f"All price sources failed: {e}", "symbol": symbol}

    async def get_company_news(self, symbol: str, days: int = 7) -> dict:
        """Get company news for last N days."""
        symbol = resolve_indian_ticker(symbol)
        # For Indian stocks, strip .NS for Finnhub news search
        finnhub_sym = symbol.replace(".NS", "").replace(".BO", "")

        try:
            loop   = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._fetch_news_sync, finnhub_sym, days),
                timeout=8.0
            )
            if result.get("news"):
                return result
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            logger.warning(f"Finnhub news error: {e}")

        # yfinance news fallback
        return await self._get_yfinance_news(symbol, finnhub_sym)

    def _fetch_news_sync(self, symbol: str, days: int) -> dict:
        try:
            end   = datetime.now()
            start = end - timedelta(days=days)
            news  = self.client.company_news(
                symbol,
                _from=start.strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d")
            )
            top_news = []
            for item in (news or [])[:5]:
                top_news.append({
                    "headline": item.get("headline"),
                    "summary":  (item.get("summary") or "")[:300],
                    "source":   item.get("source"),
                    "url":      item.get("url"),
                    "datetime": datetime.fromtimestamp(
                        item.get("datetime", 0)
                    ).strftime("%Y-%m-%d %H:%M")
                })
            return {"symbol": symbol, "news": top_news, "count": len(news or [])}
        except Exception as e:
            return {"error": str(e), "symbol": symbol, "news": []}

    async def _get_yfinance_news(self, symbol: str, finnhub_sym: str) -> dict:
        """yfinance news fallback."""
        try:
            def _fetch():
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                news = ticker.news or []
                formatted = []
                for n in news[:6]:
                    formatted.append({
                        "headline": n.get("title", ""),
                        "summary":  n.get("summary", "")[:300],
                        "source":   n.get("publisher", ""),
                        "url":      n.get("link", ""),
                        "datetime": datetime.fromtimestamp(
                            n.get("providerPublishTime", 0)
                        ).strftime("%Y-%m-%d %H:%M") if n.get("providerPublishTime") else ""
                    })
                return formatted

            loop = asyncio.get_event_loop()
            news_list = await asyncio.wait_for(
                loop.run_in_executor(None, _fetch),
                timeout=8.0
            )
            return {
                "symbol":   symbol,
                "news":     news_list,
                "count":    len(news_list),
                "_source":  "yfinance_news"
            }
        except Exception as e:
            return {"error": str(e), "symbol": symbol, "news": []}

    async def get_earnings_calendar(self, days: int = 7) -> dict:
        """Get upcoming earnings announcements."""
        try:
            loop   = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._fetch_earnings_sync, days),
                timeout=8.0
            )
            return result
        except asyncio.TimeoutError:
            return {"error": "Earnings calendar timeout"}
        except Exception as e:
            return {"error": str(e)}

    def _fetch_earnings_sync(self, days: int) -> dict:
        try:
            end      = datetime.now() + timedelta(days=days)
            calendar = self.client.earnings_calendar(
                _from=datetime.now().strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d"),
                symbol="",
                international=False
            )
            earnings = (calendar.get("earningsCalendar") or [])[:10]
            result   = []
            for e in earnings:
                result.append({
                    "symbol":           e.get("symbol"),
                    "date":             e.get("date"),
                    "hour":             e.get("hour"),
                    "eps_estimate":     e.get("epsEstimate"),
                    "revenue_estimate": e.get("revenueEstimate")
                })
            return {"earnings": result}
        except Exception as e:
            return {"error": str(e)}

    async def get_market_sentiment(self, symbol: str) -> dict:
        try:
            loop   = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.client.stock_social_sentiment(symbol)
                ),
                timeout=6.0
            )
            return result or {}
        except Exception as e:
            return {"error": str(e)}
