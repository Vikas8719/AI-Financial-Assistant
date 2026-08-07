"""Finnhub service — live prices, news, earnings calendar."""
import finnhub
from datetime import datetime, timedelta
from app.config import settings


class FinnhubService:
    def __init__(self):
        self.client = finnhub.Client(api_key=settings.FINNHUB_API_KEY)

    async def get_quote(self, symbol: str) -> dict:
        """Get real-time stock quote."""
        try:
            symbol = symbol.upper().strip()
            quote = self.client.quote(symbol)
            profile = self.client.company_profile2(symbol=symbol)
            return {
                "symbol": symbol,
                "company": profile.get("name", symbol),
                "price": quote.get("c"),
                "change": quote.get("d"),
                "change_pct": quote.get("dp"),
                "high": quote.get("h"),
                "low": quote.get("l"),
                "open": quote.get("o"),
                "prev_close": quote.get("pc"),
                "currency": profile.get("currency", "USD"),
                "exchange": profile.get("exchange", ""),
                "industry": profile.get("finnhubIndustry", ""),
                "market_cap": profile.get("marketCapitalization"),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {"error": str(e), "symbol": symbol}

    async def get_company_news(self, symbol: str, days: int = 7) -> dict:
        """Get company news for last N days."""
        try:
            symbol = symbol.upper().strip()
            end = datetime.now()
            start = end - timedelta(days=days)
            news = self.client.company_news(
                symbol,
                _from=start.strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d")
            )
            # Return top 5 most recent
            top_news = []
            for item in news[:5]:
                top_news.append({
                    "headline": item.get("headline"),
                    "summary": item.get("summary", "")[:300],
                    "source": item.get("source"),
                    "url": item.get("url"),
                    "datetime": datetime.fromtimestamp(item.get("datetime", 0)).strftime("%Y-%m-%d %H:%M")
                })
            return {"symbol": symbol, "news": top_news, "count": len(news)}
        except Exception as e:
            return {"error": str(e), "symbol": symbol}

    async def get_earnings_calendar(self, days: int = 7) -> dict:
        """Get upcoming earnings announcements."""
        try:
            end = datetime.now() + timedelta(days=days)
            calendar = self.client.earnings_calendar(
                _from=datetime.now().strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d"),
                symbol="",
                international=False
            )
            earnings = calendar.get("earningsCalendar", [])[:10]
            result = []
            for e in earnings:
                result.append({
                    "symbol": e.get("symbol"),
                    "date": e.get("date"),
                    "hour": e.get("hour"),  # bmo/amc/dmh
                    "eps_estimate": e.get("epsEstimate"),
                    "revenue_estimate": e.get("revenueEstimate")
                })
            return {"earnings": result}
        except Exception as e:
            return {"error": str(e)}

    async def get_market_sentiment(self, symbol: str) -> dict:
        """Get social sentiment for a symbol."""
        try:
            sentiment = self.client.stock_social_sentiment(symbol)
            return sentiment
        except Exception as e:
            return {"error": str(e)}
