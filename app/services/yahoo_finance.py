"""Yahoo Finance service — fundamentals, market overview, company info."""
import yfinance as yf
from typing import Optional


MARKET_INDICES = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "VIX": "^VIX"
}


class YahooFinanceService:

    async def get_fundamentals(self, symbol: str) -> dict:
        """Get company fundamentals and key financial metrics."""
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info

            return {
                "symbol": symbol.upper(),
                "name": info.get("longName") or info.get("shortName", symbol),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "country": info.get("country"),
                "description": (info.get("longBusinessSummary", "") or "")[:500],
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "revenue": info.get("totalRevenue"),
                "revenue_growth": info.get("revenueGrowth"),
                "gross_margin": info.get("grossMargins"),
                "operating_margin": info.get("operatingMargins"),
                "profit_margin": info.get("profitMargins"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "eps": info.get("trailingEps"),
                "forward_eps": info.get("forwardEps"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "free_cash_flow": info.get("freeCashflow"),
                "dividend_yield": info.get("dividendYield"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "analyst_target": info.get("targetMeanPrice"),
                "analyst_rating": info.get("recommendationKey"),
                "employees": info.get("fullTimeEmployees"),
                "website": info.get("website")
            }
        except Exception as e:
            return {"error": str(e), "symbol": symbol}

    async def get_market_overview(self) -> dict:
        """Get major indices performance."""
        result = {}
        for name, ticker_sym in MARKET_INDICES.items():
            try:
                ticker = yf.Ticker(ticker_sym)
                info = ticker.info
                hist = ticker.history(period="2d")

                if len(hist) >= 2:
                    prev_close = hist["Close"].iloc[-2]
                    current = hist["Close"].iloc[-1]
                    change = current - prev_close
                    change_pct = (change / prev_close) * 100
                else:
                    current = info.get("regularMarketPrice", 0)
                    change = info.get("regularMarketChange", 0)
                    change_pct = info.get("regularMarketChangePercent", 0)

                result[name] = {
                    "price": round(current, 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2)
                }
            except Exception:
                result[name] = {"error": "unavailable"}

        return {"indices": result}

    async def get_price_history(self, symbol: str, period: str = "1mo") -> dict:
        """Get historical price data."""
        try:
            ticker = yf.Ticker(symbol.upper())
            hist = ticker.history(period=period)
            prices = []
            for date, row in hist.iterrows():
                prices.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "close": round(row["Close"], 2),
                    "volume": int(row["Volume"])
                })
            return {"symbol": symbol, "period": period, "prices": prices[-30:]}
        except Exception as e:
            return {"error": str(e)}
