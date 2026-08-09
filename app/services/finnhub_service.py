"""
Finnhub service — live prices, news, earnings calendar.
Fix: Indian stocks auto .NS suffix + better fallback chain
"""
import asyncio
import finnhub
from datetime import datetime, timedelta
from app.config import settings

# Indian stock name → NSE ticker mapping
INDIAN_STOCKS = {
    "tata steel":   "TATASTEEL.NS",
    "tatasteel":    "TATASTEEL.NS",
    "tcs":          "TCS.NS",
    "infosys":      "INFY.NS",
    "wipro":        "WIPRO.NS",
    "hdfc":         "HDFCBANK.NS",
    "hdfcbank":     "HDFCBANK.NS",
    "reliance":     "RELIANCE.NS",
    "ril":          "RELIANCE.NS",
    "icicibank":    "ICICIBANK.NS",
    "icici":        "ICICIBANK.NS",
    "sbi":          "SBIN.NS",
    "axisbank":     "AXISBANK.NS",
    "bajajfinance": "BAJFINANCE.NS",
    "bajaj finance":"BAJFINANCE.NS",
    "itc":          "ITC.NS",
    "maruti":       "MARUTI.NS",
    "tatamotors":   "TATAMOTORS.NS",
    "tata motors":  "TATAMOTORS.NS",
    "sunpharma":    "SUNPHARMA.NS",
    "sun pharma":   "SUNPHARMA.NS",
    "hcltech":      "HCLTECH.NS",
    "hcl":          "HCLTECH.NS",
    "ultracemco":   "ULTRACEMCO.NS",
    "asianpaint":   "ASIANPAINT.NS",
    "asian paints": "ASIANPAINT.NS",
    "kotak":        "KOTAKBANK.NS",
    "kotakbank":    "KOTAKBANK.NS",
    "bhartiairtel": "BHARTIAIRTEL.NS",
    "airtel":       "BHARTIAIRTEL.NS",
    "adanient":     "ADANIENT.NS",
    "adani":        "ADANIENT.NS",
    "adaniports":   "ADANIPORTS.NS",
    "ntpc":         "NTPC.NS",
    "ongc":         "ONGC.NS",
    "powergrid":    "POWERGRID.NS",
    "titan":        "TITAN.NS",
    "nestleindia":  "NESTLEIND.NS",
    "nestle":       "NESTLEIND.NS",
    "lti":          "LTIM.NS",
    "ltimindtree":  "LTIM.NS",
    "drreddy":      "DRREDDY.NS",
    "dr reddy":     "DRREDDY.NS",
    "cipla":        "CIPLA.NS",
    "divislab":     "DIVISLAB.NS",
    "hindalco":     "HINDALCO.NS",
    "jswsteel":     "JSWSTEEL.NS",
    "jsw steel":    "JSWSTEEL.NS",
    "bajaj auto":   "BAJAJ-AUTO.NS",
    "bajajauto":    "BAJAJ-AUTO.NS",
    "eicher":       "EICHERMOT.NS",
    "hero moto":    "HEROMOTOCO.NS",
    "heromoto":     "HEROMOTOCO.NS",
    "indusindbk":   "INDUSINDBK.NS",
    "indusind":     "INDUSINDBK.NS",
}


def resolve_indian_ticker(symbol: str) -> str:
    """
    Resolve Indian company name → NSE ticker.
    'Tata Steel' → 'TATASTEEL.NS'
    'TATASTEEL'  → 'TATASTEEL.NS' (auto-add .NS if looks Indian)
    """
    key = symbol.strip().lower().replace(" ", "")
    # Direct name match
    for name, ticker in INDIAN_STOCKS.items():
        if key == name.replace(" ", ""):
            return ticker
    # If already has .NS or .BO — return as is
    upper = symbol.upper().strip()
    if ".NS" in upper or ".BO" in upper:
        return upper
    # If it's a known Finnhub-listed US ticker — return as is
    us_tickers = {"AAPL","MSFT","GOOGL","GOOG","META","AMZN","NVDA","TSLA",
                  "NFLX","AMD","INTC","CRM","ORCL","IBM","QCOM","BABA","TSM"}
    if upper in us_tickers:
        return upper
    return symbol.upper().strip()


class FinnhubService:
    def __init__(self):
        self.client = finnhub.Client(api_key=settings.FINNHUB_API_KEY)

    async def get_quote(self, symbol: str) -> dict:
        """
        Get real-time stock quote.
        Indian stocks: auto-try .NS suffix via Yahoo fallback.
        """
        symbol = resolve_indian_ticker(symbol)

        # For .NS stocks Finnhub doesn't cover — use Yahoo directly
        if ".NS" in symbol or ".BO" in symbol:
            return await self._get_indian_quote(symbol)

        try:
            loop    = asyncio.get_event_loop()
            result  = await asyncio.wait_for(
                loop.run_in_executor(None, self._fetch_quote_sync, symbol),
                timeout=8.0
            )
            return result
        except asyncio.TimeoutError:
            return {"error": "Finnhub timeout", "symbol": symbol}
        except Exception as e:
            return {"error": str(e), "symbol": symbol}

    def _fetch_quote_sync(self, symbol: str) -> dict:
        try:
            quote   = self.client.quote(symbol)
            profile = {}
            try:
                profile = self.client.company_profile2(symbol=symbol) or {}
            except Exception:
                pass

            price = quote.get("c")
            if not price:
                return {"error": "No price data", "symbol": symbol}

            return {
                "symbol":     symbol,
                "company":    profile.get("name", symbol),
                "price":      price,
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
                "timestamp":  datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {"error": str(e), "symbol": symbol}

    async def _get_indian_quote(self, symbol: str) -> dict:
        """Yahoo Finance fallback for Indian NSE/BSE stocks."""
        try:
            import yfinance as yf
            loop   = asyncio.get_event_loop()

            def _fetch():
                ticker = yf.Ticker(symbol)
                info   = ticker.info or {}
                price  = (info.get("currentPrice")
                          or info.get("regularMarketPrice")
                          or info.get("previousClose"))
                if not price:
                    hist = ticker.history(period="1d", timeout=5)
                    if hist is not None and len(hist) > 0:
                        price = float(hist["Close"].iloc[-1])
                return {
                    "symbol":     symbol,
                    "company":    info.get("longName") or info.get("shortName", symbol),
                    "price":      price,
                    "change":     info.get("regularMarketChange"),
                    "change_pct": info.get("regularMarketChangePercent"),
                    "high":       info.get("dayHigh"),
                    "low":        info.get("dayLow"),
                    "open":       info.get("open"),
                    "prev_close": info.get("previousClose"),
                    "currency":   info.get("currency", "INR"),
                    "exchange":   info.get("exchange", "NSE"),
                    "market_cap": info.get("marketCap"),
                    "_source":    "yahoo_india",
                    "timestamp":  datetime.utcnow().isoformat()
                }

            result = await asyncio.wait_for(
                loop.run_in_executor(None, _fetch),
                timeout=10.0
            )
            if result.get("price"):
                return result
            return {"error": "No price data for Indian stock", "symbol": symbol}

        except Exception as e:
            return {"error": str(e), "symbol": symbol}

    async def get_company_news(self, symbol: str, days: int = 7) -> dict:
        """Get company news for last N days."""
        symbol = resolve_indian_ticker(symbol)
        # For Indian stocks strip .NS for Finnhub news search
        finnhub_sym = symbol.replace(".NS", "").replace(".BO", "")

        try:
            loop   = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._fetch_news_sync, finnhub_sym, days),
                timeout=8.0
            )
            return result
        except asyncio.TimeoutError:
            return {"error": "News fetch timeout", "symbol": symbol}
        except Exception as e:
            return {"error": str(e), "symbol": symbol}

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
            return {"error": str(e), "symbol": symbol}

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
                    "symbol":            e.get("symbol"),
                    "date":              e.get("date"),
                    "hour":              e.get("hour"),
                    "eps_estimate":      e.get("epsEstimate"),
                    "revenue_estimate":  e.get("revenueEstimate")
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
