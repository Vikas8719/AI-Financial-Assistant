"""Web search service — Tavily primary, DuckDuckGo fallback."""
from app.config import settings
from typing import Optional
import httpx


class WebSearchService:

    async def search(self, query: str, max_results: int = 5) -> dict:
        """Search the web for financial information."""
        # Try Tavily first (higher quality)
        if settings.TAVILY_API_KEY:
            result = await self._tavily_search(query, max_results)
            if result and not result.get("error"):
                return result

        # Fallback to DuckDuckGo
        return await self._ddg_search(query, max_results)

    async def _tavily_search(self, query: str, max_results: int = 5) -> dict:
        """Tavily AI-powered search."""
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
                            "finance.yahoo.com", "businessinsider.com", "techcrunch.com"
                        ]
                    },
                    timeout=15.0
                )
                data = response.json()
                results = []
                for r in data.get("results", []):
                    results.append({
                        "title": r.get("title"),
                        "url": r.get("url"),
                        "snippet": r.get("content", "")[:400],
                        "score": r.get("score")
                    })
                return {
                    "query": query,
                    "answer": data.get("answer", ""),
                    "results": results,
                    "source": "tavily"
                }
        except Exception as e:
            return {"error": str(e)}

    async def _ddg_search(self, query: str, max_results: int = 5) -> dict:
        """DuckDuckGo search fallback."""
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    f"{query} finance",
                    max_results=max_results,
                    safesearch="moderate"
                ))
            formatted = []
            for r in results:
                formatted.append({
                    "title": r.get("title"),
                    "url": r.get("href"),
                    "snippet": r.get("body", "")[:400]
                })
            return {"query": query, "results": formatted, "source": "duckduckgo"}
        except Exception as e:
            return {"error": str(e), "results": []}
