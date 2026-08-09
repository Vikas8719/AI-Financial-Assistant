"""
Yahoo Finance service — fundamentals, market overview, company info.
Fix: get_market_overview timeout + better error handling
"""
import asyncio
import yfinance as yf
from typing import Optional


MARKET_INDICES = {
    "S&P 500":  "^GSPC",
    "NASDAQ":   "^IXIC",
    "Dow Jones":"^DJI",
    "Nifty 50": "^NSEI",
    "Sensex":   "^BSESN",
    "VIX":      "^VIX"
}


def _safe_fetch_index(ticker_sym: str) -> dict:
    """Fetch single index — blocking, runs in thread pool."""
    try:
        ticker = yf.Ticker(ticker_sym)
        hist   = ticker.history(period="2d", timeout=6)
        if hist is not None and len(hist) >= 2:
            prev    = hist["Close"].iloc[-2]
            current = hist["Close"].iloc[-1]
            change  = current - prev
            return {
                "price":      round(float(current), 2),
                "change":     round(float(change), 2),
                "change_pct": round(float(change / prev * 100), 2)
            }
        # fallback to info
        info    = ticker.info or {}
        current = info.get("regularMarketPrice") or info.get("previousClose", 0)
        change  = info.get("regularMarketChange", 0)
        pct     = info.get("regularMarketChangePercent", 0)
        if current:
            return {
                "price":      round(float(current), 2),
                "change":     round(float(change), 2),
                "change_pct": round(float(pct), 2)
            }
    except Exception:
        pass
    return {"error": "unavailable"}


def _safe_fetch_ticker(symbol: str) -> dict:
    """Fetch ticker fundamentals — blocking."""
    try:
        ticker = yf.Ticker(symbol.upper())
        info   = ticker.info or {}
        if not info.get("symbol") and not info.get("longName"):
            return {"error": "no data", "symbol": symbol}
        return {
            "symbol":           symbol.upper(),
            "name":             info.get("longName") or info.get("shortName", symbol),
            "sector":           info.get("sector"),
            "industry":         info.get("industry"),
            "country":          info.get("country"),
            "description":      (info.get("longBusinessSummary") or "")[:500],
            "market_cap":       info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "revenue":          info.get("totalRevenue"),
            "revenue_growth":   info.get("revenueGrowth"),
            "gross_margin":     info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "profit_margin":    info.get("profitMargins"),
            "pe_ratio":         info.get("trailingPE"),
            "forward_pe":       info.get("forwardPE"),
            "peg_ratio":        info.get("pegRatio"),
            "price_to_book":    info.get("priceToBook"),
            "eps":              info.get("trailingEps"),
            "forward_eps":      info.get("forwardEps"),
            "debt_to_equity":   info.get("debtToEquity"),
            "current_ratio":    info.get("currentRatio"),
            "free_cash_flow":   info.get("freeCashflow"),
            "dividend_yield":   info.get("dividendYield"),
            "52w_high":         info.get("fiftyTwoWeekHigh"),
            "52w_low":          info.get("fiftyTwoWeekLow"),
            "analyst_target":   info.get("targetMeanPrice"),
            "analyst_rating":   info.get("recommendationKey"),
            "employees":        info.get("fullTimeEmployees"),
            "website":          info.get("website"),
            "current_price":    info.get("currentPrice") or info.get("regularMarketPrice"),
            "change_pct":       info.get("regularMarketChangePercent"),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


class YahooFinanceService:

    async def get_fundamentals(self, symbol: str) -> dict:
        """Get company fundamentals — runs yfinance in thread pool (non-blocking)."""
        try:
            loop   = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _safe_fetch_ticker, symbol),
                timeout=10.0
            )
            return result
        except asyncio.TimeoutError:
            return {"error": "Yahoo Finance timeout", "symbol": symbol}
        except Exception as e:
            return {"error": str(e), "symbol": symbol}

    async def get_market_overview(self) -> dict:
        """
        Get major indices — all fetched concurrently with per-index timeout.
        Fixed: was blocking event loop, causing 'Something went wrong' error.
        """
        loop = asyncio.get_event_loop()

        async def _fetch_one(name: str, sym: str) -> tuple[str, dict]:
            try:
                data = await asyncio.wait_for(
                    loop.run_in_executor(None, _safe_fetch_index, sym),
                    timeout=8.0
                )
                return name, data
            except asyncio.TimeoutError:
                return name, {"error": "timeout"}
            except Exception as e:
                return name, {"error": str(e)}

        # All 6 indices fetched concurrently
        tasks   = [_fetch_one(n, s) for n, s in MARKET_INDICES.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        indices = {}
        for r in results:
            if isinstance(r, tuple):
                name, data = r
                indices[name] = data

        # Check if we got at least some data
        working = {k: v for k, v in indices.items() if not v.get("error")}
        if not working:
            return {"error": "All market data sources unavailable", "indices": indices}

        return {"indices": indices}

    async def get_price_history(self, symbol: str, period: str = "1mo") -> dict:
        """Get historical price data."""
        def _fetch():
            try:
                ticker = yf.Ticker(symbol.upper())
                hist   = ticker.history(period=period, timeout=8)
                prices = []
                for date, row in hist.iterrows():
                    prices.append({
                        "date":   date.strftime("%Y-%m-%d"),
                        "close":  round(float(row["Close"]), 2),
                        "volume": int(row["Volume"])
                    })
                return {"symbol": symbol, "period": period, "prices": prices[-30:]}
            except Exception as e:
                return {"error": str(e)}

        try:
            loop   = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _fetch),
                timeout=10.0
            )
            return result
        except asyncio.TimeoutError:
            return {"error": "Timeout fetching price history"}
        except Exception as e:
            return {"error": str(e)}
