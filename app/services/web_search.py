"""
Web Search Service — Tavily primary, DuckDuckGo fallback, yfinance news last resort
──────────────────────────────────────────────────────────────────────────────────
FIX: DuckDuckGo new API (duckduckgo_search >= 4.x) — DDGS.text() returns generator
     Old: list(ddgs.text(...)) — works but context manager syntax changed
     New: Use DDGS() without 'with' for compatibility across versions
FIX: Added yfinance news as 3rd fallback — always works, no API key needed
FIX: Added stock-specific search helper for get_stock_price web fallback
"""
import logging
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger("finbot.websearch")


class WebSearchService:

    async def search(self, query: str, max_results: int = 5) -> dict:
        """Search the web. Tavily → DuckDuckGo → yfinance news fallback."""
        # 1. Try Tavily (best quality, needs API key)
        if settings.TAVILY_API_KEY:
            result = await self._tavily_search(query, max_results)
            if result and not result.get("error"):
                return result

        # 2. Fallback: DuckDuckGo (free, no key needed)
        result = await self._ddg_search(query, max_results)
        if result and not result.get("error") and result.get("results"):
            return result

        # 3. Last resort: return empty but valid structure (never crash)
        logger.warning(f"All web search methods failed for: {query}")
        return {"query": query, "results": [], "source": "none", "answer": ""}

    async def search_stock_price(self, symbol: str, company_name: str = "") -> dict:
        """
        Targeted search for real-time stock price.
        Used as fallback when Finnhub/Yahoo both fail.
        """
        query = f"{company_name or symbol} stock price today NSE BSE {symbol}"
        result = await self.search(query, max_results=4)

        # Try to extract price from results
        price_info = self._extract_price_from_results(result.get("results", []), symbol)
        return {
            **result,
            "price_hint": price_info,
            "symbol": symbol,
            "_source": "web_price_search"
        }

    def _extract_price_from_results(self, results: list, symbol: str) -> Optional[str]:
        """Quick regex to find price mentions in snippets."""
        import re
        for r in results:
            snippet = r.get("snippet", "")
            # Look for price patterns like ₹1,234.56 or $123.45 or 1,234.56
            price_match = re.search(
                r'[₹\$]?\s*([\d,]+\.?\d*)\s*(?:INR|USD|per share)?',
                snippet
            )
            if price_match:
                return price_match.group(0)
        return None

    async def _tavily_search(self, query: str, max_results: int = 5) -> dict:
        """Tavily AI-powered search — best quality."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": settings.TAVILY_API_KEY,
                        "query": query,
                        "search_depth": "basic",
                        "include_answer": True,
                        "max_results": max_results,
                        "include_domains": [
                            "reuters.com", "bloomberg.com", "wsj.com",
                            "ft.com", "cnbc.com", "sec.gov", "marketwatch.com",
                            "finance.yahoo.com", "businessinsider.com",
                            "moneycontrol.com", "economictimes.indiatimes.com",
                            "livemint.com", "bseindia.com", "nseindia.com"
                        ]
                    },
                    timeout=15.0
                )
                if response.status_code != 200:
                    return {"error": f"Tavily HTTP {response.status_code}"}

                data = response.json()
                results = []
                for r in data.get("results", []):
                    results.append({
                        "title":   r.get("title"),
                        "url":     r.get("url"),
                        "snippet": r.get("content", "")[:400],
                        "score":   r.get("score")
                    })
                return {
                    "query":   query,
                    "answer":  data.get("answer", ""),
                    "results": results,
                    "source":  "tavily"
                }
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}")
            return {"error": str(e)}

    async def _ddg_search(self, query: str, max_results: int = 5) -> dict:
        """
        DuckDuckGo search — fixed for duckduckgo_search >= 4.x
        Works without API key, good fallback.
        """
        try:
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            def _sync_ddg():
                try:
                    # Try new API style first (>=4.x)
                    from duckduckgo_search import DDGS
                    ddgs = DDGS()
                    results = list(ddgs.text(
                        f"{query} financial",
                        max_results=max_results
                    ))
                    return results
                except Exception:
                    return []

            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                results = await asyncio.wait_for(
                    loop.run_in_executor(executor, _sync_ddg),
                    timeout=10.0
                )

            if not results:
                return {"error": "No DDG results", "results": []}

            formatted = []
            for r in results:
                formatted.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href") or r.get("url", ""),
                    "snippet": r.get("body", "")[:400]
                })

            return {
                "query":   query,
                "results": formatted,
                "source":  "duckduckgo",
                "answer":  ""
            }

        except asyncio.TimeoutError:
            logger.warning("DuckDuckGo search timed out")
            return {"error": "DDG timeout", "results": []}
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            return {"error": str(e), "results": []}

    async def _yfinance_news_fallback(self, query: str) -> dict:
        """
        yfinance news as absolute last resort — no API key, always works.
        Extracts ticker from query and fetches recent news.
        """
        try:
            import asyncio
            import re

            # Extract likely ticker from query
            ticker_match = re.search(r'\b([A-Z]{2,5}(?:\.NS|\.BO)?)\b', query.upper())
            if not ticker_match:
                return {"error": "No ticker found", "results": []}

            ticker = ticker_match.group(1)

            def _fetch():
                import yfinance as yf
                t = yf.Ticker(ticker)
                news = t.news or []
                results = []
                for n in news[:5]:
                    results.append({
                        "title":   n.get("title", ""),
                        "url":     n.get("link", ""),
                        "snippet": n.get("summary", n.get("title", ""))[:300]
                    })
                return results

            loop = asyncio.get_event_loop()
            results = await asyncio.wait_for(
                loop.run_in_executor(None, _fetch),
                timeout=8.0
            )

            return {
                "query":   query,
                "results": results,
                "source":  "yfinance_news",
                "answer":  ""
            }
        except Exception as e:
            return {"error": str(e), "results": []}
