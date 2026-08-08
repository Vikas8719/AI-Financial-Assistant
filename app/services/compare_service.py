"""
Smart Compare Service — Yahoo → Finnhub → Web Search fallback chain
────────────────────────────────────────────────────────────────────
Problem: compare_companies sirf Yahoo call karta tha — agar Yahoo se
         data na mile (MSFT, GOOGL kabhi kabhi fail karte hain) to
         agent "encountered an error" deta tha.

Solution: 3-layer fallback per company, phir BM25+reranker se
          jo bhi mila usse 70b ko deta hai synthesis ke liye.
          User ko kabhi error nahi dikhta.

Fallback chain per company:
  Layer 1 → Yahoo Finance (yfinance)
  Layer 2 → Finnhub quote + profile
  Layer 3 → Web search (Tavily/DDG) + BM25 parse
"""
import asyncio
import re
from typing import Optional

from app.services.yahoo_finance import YahooFinanceService
from app.services.finnhub_service import FinnhubService
from app.services.web_search import WebSearchService
from app.ai.rag_engine import tokenize_financial, bm25_score, chunk_document
from app.ai.reranker import rerank_chunks

yahoo   = YahooFinanceService()
finnhub = FinnhubService()
search  = WebSearchService()


# ── known ticker aliases ──────────────────────────────────────────────────────
TICKER_ALIASES = {
    "tesla":     "TSLA",  "tsla":    "TSLA",
    "apple":     "AAPL",  "aapl":    "AAPL",
    "microsoft": "MSFT",  "msft":    "MSFT",
    "google":    "GOOGL", "googl":   "GOOGL", "alphabet": "GOOGL",
    "meta":      "META",  "facebook":"META",  "fb":       "META",
    "amazon":    "AMZN",  "amzn":    "AMZN",
    "nvidia":    "NVDA",  "nvda":    "NVDA",
    "netflix":   "NFLX",  "nflx":    "NFLX",
    "reliance":  "RELIANCE.NS",
    "tcs":       "TCS.NS",
    "infosys":   "INFY",
    "hdfc":      "HDFCBANK.NS",
    "wipro":     "WIPRO.NS",
    "samsung":   "005930.KS",
    "alibaba":   "BABA",
    "baba":      "BABA",
    "tsmc":      "TSM",
}

# Fields we want per company for comparison
COMPARE_FIELDS = [
    "name", "symbol", "sector", "market_cap", "revenue",
    "pe_ratio", "forward_pe", "eps", "profit_margin",
    "operating_margin", "revenue_growth", "debt_to_equity",
    "free_cash_flow", "dividend_yield", "52w_high", "52w_low",
    "analyst_rating", "analyst_target", "employees", "description"
]


def resolve_ticker(name_or_ticker: str) -> str:
    """
    Resolve company name or partial ticker to canonical ticker symbol.
    'tesla' → 'TSLA', 'Microsoft' → 'MSFT', 'AAPL' → 'AAPL'
    """
    key = name_or_ticker.strip().lower()
    if key in TICKER_ALIASES:
        return TICKER_ALIASES[key]
    return name_or_ticker.strip().upper()


def _is_data_sufficient(data: dict) -> bool:
    """Check if we got at least some useful fields (not just error or empty)."""
    if data.get("error"):
        return False
    useful = ["market_cap", "revenue", "pe_ratio", "eps", "profit_margin", "name"]
    return any(data.get(f) for f in useful)


def _is_price_sufficient(data: dict) -> bool:
    """Check if Finnhub quote has useful data."""
    return bool(data.get("price") or data.get("market_cap"))


# ── Layer 1: Yahoo Finance ────────────────────────────────────────────────────

async def _fetch_yahoo(symbol: str) -> dict:
    try:
        data = await yahoo.get_fundamentals(symbol)
        return data
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


# ── Layer 2: Finnhub (quote + company profile) ───────────────────────────────

async def _fetch_finnhub(symbol: str) -> dict:
    """
    Finnhub gives: price, change, 52w high/low, market cap, company name.
    Not as rich as Yahoo but reliable.
    """
    try:
        quote   = await finnhub.get_quote(symbol)
        profile = {}
        try:
            # Finnhub company profile endpoint
            raw = finnhub.client.company_profile2(symbol=symbol)
            profile = raw or {}
        except Exception:
            pass

        if quote.get("error") and not profile:
            return {"error": "Finnhub returned no data", "symbol": symbol}

        return {
            "symbol":       symbol,
            "name":         profile.get("name") or quote.get("company") or symbol,
            "sector":       profile.get("finnhubIndustry"),
            "market_cap":   profile.get("marketCapitalization"),   # in millions
            "pe_ratio":     quote.get("pe"),
            "52w_high":     quote.get("high52w") or profile.get("52WeekHigh"),
            "52w_low":      quote.get("low52w")  or profile.get("52WeekLow"),
            "price":        quote.get("price"),
            "change_pct":   quote.get("change_pct"),
            "employees":    profile.get("employeeTotal"),
            "website":      profile.get("weburl"),
            "description":  profile.get("name", ""),
            "country":      profile.get("country"),
            "exchange":     profile.get("exchange"),
            "_source":      "finnhub"
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


# ── Layer 3: Web Search + BM25 parse ─────────────────────────────────────────

async def _fetch_web(company_name: str, symbol: str) -> dict:
    """
    Web search for company financials.
    Uses BM25 to extract relevant numbers from search snippets.
    Reranker picks the most relevant snippet.
    """
    try:
        query  = f"{company_name} {symbol} revenue earnings PE ratio market cap 2024 financials"
        result = await search.search(query, max_results=6)

        snippets = []
        for r in result.get("results", []):
            text = f"{r.get('title','')} {r.get('snippet','')}"
            if text.strip():
                snippets.append(text)

        if not snippets:
            return {"error": "No web results", "symbol": symbol, "_source": "web_failed"}

        # Join all snippets into a pseudo-document
        combined = "\n\n".join(snippets)

        # BM25 over snippets to find most financially relevant
        query_tokens = tokenize_financial(
            f"revenue earnings profit pe ratio market cap {company_name}"
        )
        chunks       = chunk_document(combined, chunk_size=120, overlap=20)

        if chunks:
            # Quick BM25 scoring
            from collections import Counter
            import math
            doc_freq: dict = {}
            for c in chunks:
                for t in set(c["tokens"]):
                    doc_freq[t] = doc_freq.get(t, 0) + 1
            avg_len = sum(len(c["tokens"]) for c in chunks) / max(len(chunks), 1)

            for c in chunks:
                score = bm25_score(
                    query_tokens, c["tokens"], doc_freq, len(chunks), avg_len
                )
                c["bm25_score"] = score

            # Rerank top chunks with 8b model
            top = sorted(chunks, key=lambda x: x["bm25_score"], reverse=True)[:8]
            if top:
                reranked = await rerank_chunks(
                    query=f"{company_name} revenue earnings market cap",
                    chunks=top,
                    top_k=3,
                    context_hint=f"extracting financial data for {company_name}"
                )
                best_text = " ".join(c["text"] for c in reranked)
            else:
                best_text = combined[:600]
        else:
            best_text = combined[:600]

        # Extract numbers from text for structured display
        parsed = _parse_financials_from_text(best_text, company_name, symbol)
        parsed["_source"]      = "web_search"
        parsed["_raw_snippet"] = best_text[:400]
        return parsed

    except Exception as e:
        return {"error": str(e), "symbol": symbol, "_source": "web_failed"}


def _parse_financials_from_text(text: str, company: str, symbol: str) -> dict:
    """
    Simple regex extraction of key financial figures from web text.
    Not perfect but gives LLM structured hints.
    """
    data = {"symbol": symbol, "name": company}

    # Market cap patterns: "$2.8 trillion", "$500 billion", "$45B"
    mc = re.search(
        r'market\s*cap[a-z\s]*[\$:]\s*([\d.]+)\s*(trillion|billion|million|[TBM])',
        text, re.IGNORECASE
    )
    if mc:
        val, unit = float(mc.group(1)), mc.group(2).lower()
        mult = {"trillion": 1e12, "billion": 1e9, "million": 1e6, "t": 1e12, "b": 1e9, "m": 1e6}
        data["market_cap"] = int(val * mult.get(unit, 1e9))

    # Revenue
    rev = re.search(
        r'revenue[a-z\s]*[\$:]\s*([\d.]+)\s*(trillion|billion|million|[TBM])',
        text, re.IGNORECASE
    )
    if rev:
        val, unit = float(rev.group(1)), rev.group(2).lower()
        mult = {"trillion": 1e12, "billion": 1e9, "million": 1e6, "t": 1e12, "b": 1e9, "m": 1e6}
        data["revenue"] = int(val * mult.get(unit, 1e9))

    # P/E ratio
    pe = re.search(r'p/?e\s*ratio[:\s]*([\d.]+)', text, re.IGNORECASE)
    if pe:
        data["pe_ratio"] = float(pe.group(1))

    # Employees
    emp = re.search(r'([\d,]+)\s*(?:full[- ]time\s*)?employees', text, re.IGNORECASE)
    if emp:
        data["employees"] = int(emp.group(1).replace(",", ""))

    return data


# ── Main compare function ─────────────────────────────────────────────────────

async def smart_compare(symbol1_raw: str, symbol2_raw: str) -> dict:
    """
    Smart company comparison with 3-layer fallback per company.
    Never returns an error to the user — always returns best available data.

    Returns dict with keys: company1, company2, sources, partial
    """
    sym1 = resolve_ticker(symbol1_raw)
    sym2 = resolve_ticker(symbol2_raw)

    # Fetch both companies concurrently — all 3 layers per company
    async def _fetch_with_fallback(symbol: str, name_hint: str) -> dict:
        # Layer 1: Yahoo
        data = await _fetch_yahoo(symbol)
        if _is_data_sufficient(data):
            data["_source"] = "yahoo"
            return data

        # Layer 2: Finnhub
        data2 = await _fetch_finnhub(symbol)
        if _is_price_sufficient(data2):
            # Merge Yahoo partial data + Finnhub
            merged = {**data, **{k: v for k, v in data2.items() if v is not None}}
            merged["_source"] = "finnhub"
            return merged

        # Layer 3: Web search + BM25 + reranker
        data3 = await _fetch_web(name_hint, symbol)
        if not data3.get("error"):
            merged = {**data, **data2, **{k: v for k, v in data3.items() if v is not None}}
            merged["_source"] = "web_search"
            return merged

        # All layers failed — return whatever partial we have
        partial = {**data, **data2, "symbol": symbol, "name": name_hint,
                   "_source": "partial", "_all_failed": True}
        return partial

    c1_data, c2_data = await asyncio.gather(
        _fetch_with_fallback(sym1, symbol1_raw),
        _fetch_with_fallback(sym2, symbol2_raw)
    )

    return {
        "company1": c1_data,
        "company2": c2_data,
        "symbols":  [sym1, sym2],
        "sources":  [c1_data.get("_source", "unknown"), c2_data.get("_source", "unknown")]
    }


async def smart_compare_multi(symbols_raw: list[str]) -> dict:
    """
    Compare 3+ companies — extends smart_compare for multi-company queries.
    e.g. "compare Tesla, Google, and Microsoft"
    """
    async def _fetch(sym_raw: str) -> dict:
        sym  = resolve_ticker(sym_raw)

        data = await _fetch_yahoo(sym)
        if _is_data_sufficient(data):
            data["_source"] = "yahoo"
            return data

        data2 = await _fetch_finnhub(sym)
        if _is_price_sufficient(data2):
            data2["_source"] = "finnhub"
            return data2

        data3 = await _fetch_web(sym_raw, sym)
        data3["_source"] = data3.get("_source", "web_search")
        return data3

    results = await asyncio.gather(*[_fetch(s) for s in symbols_raw])
    return {
        f"company{i+1}": r
        for i, r in enumerate(results)
    }
