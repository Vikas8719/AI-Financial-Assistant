"""
Research Engine — Deep multi-source company research
─────────────────────────────────────────────────────────────────────
Jab user "research X" ya "deep dive into Y" likhe:

Pipeline:
  Step 1 → Sources collect karo concurrently (5-6 parallel calls)
           • Yahoo fundamentals
           • Finnhub news (last 30 days)
           • SEC filings (10-K / 10-Q)
           • Web search (3 targeted queries)
           • Finnhub profile + peers
  Step 2 → BM25 over all collected text → top chunks
  Step 3 → 8b reranker → best 8 chunks
  Step 4 → 70b synthesizes structured research report

Output format (Telegram-friendly):
  📊 Company snapshot
  💰 Financials (key metrics)
  📰 Recent news (top 3)
  ⚠️  Key risks
  🔭 Outlook
  🔗 Sources used
"""
import asyncio
import json
import logging
from typing import Optional

from groq import AsyncGroq

from app.config import settings
from app.services.yahoo_finance import YahooFinanceService
from app.services.finnhub_service import FinnhubService
from app.services.web_search import WebSearchService
from app.services.sec_edgar import SecEdgarService
from app.services.compare_service import resolve_ticker
from app.ai.rag_engine import VectorlessRAG, chunk_document, tokenize_financial, bm25_score
from app.ai.reranker import rerank_chunks

logger = logging.getLogger("finbot.research")

yahoo   = YahooFinanceService()
finnhub = FinnhubService()
search  = WebSearchService()
sec     = SecEdgarService()
client  = AsyncGroq(api_key=settings.GROQ_API_KEY)

RESEARCH_MODEL   = "openai/gpt-oss-120b"
RESEARCH_TEMP    = 0.15   # Near-zero — factual synthesis, no hallucination


# ──────────────────────────────────────────────
#  Research intent detector
# ──────────────────────────────────────────────

RESEARCH_TRIGGERS = [
    "research", "deep dive", "deep-dive", "full analysis",
    "detailed analysis", "analyze", "analyse", "tell me everything",
    "complete overview", "comprehensive", "in depth", "in-depth",
    "full report", "breakdown", "break down", "give me all",
    "everything about", "all about", "deep analysis", "due diligence",
    "dd on", "thesis on", "investment thesis", "deep research"
]

def is_research_query(text: str) -> tuple[bool, str]:
    """
    Returns (is_research, company_name_or_ticker).
    Detects: "research Tesla", "deep dive AAPL", "analyze Microsoft"
    """
    text_lower = text.lower().strip()

    for trigger in RESEARCH_TRIGGERS:
        if trigger in text_lower:
            # Extract company name after the trigger
            idx   = text_lower.find(trigger)
            after = text[idx + len(trigger):].strip()

            # Remove common filler words
            after = after.lstrip("on into for about :- ").strip()

            # Clean up
            company = after.split("\n")[0].strip()
            if company:
                return True, company

    return False, ""


# ──────────────────────────────────────────────
#  Data collectors — all concurrent
# ──────────────────────────────────────────────

async def _collect_fundamentals(symbol: str) -> dict:
    try:
        data = await yahoo.get_fundamentals(symbol)
        if not data.get("error"):
            return data
    except Exception:
        pass
    # Finnhub fallback
    try:
        from app.services.compare_service import _fetch_finnhub
        return await _fetch_finnhub(symbol)
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


async def _collect_news(symbol: str) -> list:
    try:
        result = await finnhub.get_company_news(symbol, days=30)
        news = result.get("news", [])
        if news:
            return news[:8]
    except Exception:
        pass
    # Web fallback
    try:
        ws = await search.search(f"{symbol} company news latest 2024 2025")
        return ws.get("results", [])[:6]
    except Exception:
        return []


async def _collect_sec(company_name: str) -> dict:
    try:
        return await sec.search_filings(company_name, "10-K")
    except Exception:
        return {}


async def _collect_web(symbol: str, company_name: str) -> list[dict]:
    """3 targeted web searches for richer research context."""
    queries = [
        f"{company_name} {symbol} business model revenue growth 2024 2025",
        f"{company_name} risks challenges competitors market position",
        f"{company_name} outlook forecast analyst opinion future"
    ]
    all_results = []
    try:
        results = await asyncio.gather(*[search.search(q, max_results=4) for q in queries])
        for r in results:
            all_results.extend(r.get("results", []))
    except Exception:
        pass
    return all_results


async def _collect_peers(symbol: str) -> list:
    """Get peer/competitor symbols from Finnhub."""
    try:
        peers = finnhub.client.company_peers(symbol)
        return peers[:5] if peers else []
    except Exception:
        return []


# ──────────────────────────────────────────────
#  BM25 + Reranker over collected text
# ──────────────────────────────────────────────

async def _rag_over_research(query: str, raw_text: str, top_k: int = 8) -> str:
    """
    Apply BM25 + 8b reranker over all collected research text.
    Returns top relevant chunks as context for 70b synthesis.
    """
    if not raw_text or len(raw_text) < 100:
        return raw_text[:4000]

    chunks = chunk_document(raw_text, chunk_size=300, overlap=60)
    if not chunks:
        return raw_text[:4000]

    # BM25 scoring
    query_tokens = tokenize_financial(query)
    doc_freq: dict = {}
    for c in chunks:
        for t in set(c["tokens"]):
            doc_freq[t] = doc_freq.get(t, 0) + 1

    avg_len = sum(len(c["tokens"]) for c in chunks) / max(len(chunks), 1)
    for c in chunks:
        c["bm25_score"] = bm25_score(
            query_tokens, c["tokens"], doc_freq, len(chunks), avg_len
        )

    # Top 15 by BM25 → reranker
    top15 = sorted(chunks, key=lambda x: x["bm25_score"], reverse=True)[:15]

    reranked = await rerank_chunks(
        query=query,
        chunks=top15,
        top_k=top_k,
        context_hint=f"deep research on {query}"
    )

    # Reassemble ordered by doc position
    ordered = sorted(reranked, key=lambda c: c.get("id", 0))
    return "\n\n".join(c["text"] for c in ordered)


# ──────────────────────────────────────────────
#  Research report generator
# ──────────────────────────────────────────────

RESEARCH_PROMPT = """You are an elite financial research analyst.
Generate a comprehensive but concise research report for Telegram (no markdown tables).

Company: {company}
User profile: {user_profile}

## DATA AVAILABLE:

### Fundamentals:
{fundamentals}

### Recent News (last 30 days):
{news}

### SEC Filing Info:
{sec_info}

### Research Context (web + BM25 + AI reranked):
{rag_context}

### Peer Companies:
{peers}

## REPORT FORMAT:
Write in this exact structure (use Telegram-friendly formatting):

📊 **{company} — Research Brief**

**🏢 Business Overview**
[2-3 sentences: what they do, market position, key segments]

**💰 Financial Snapshot**
[Key metrics with actual numbers: revenue, market cap, margins, PE, growth]

**📈 Recent Developments**
[Top 2-3 news items that actually matter — with dates]

**⚠️ Key Risks**
[3-4 specific, real risks — not generic]

**🔭 Outlook**
[2-3 sentences: near-term catalysts, analyst consensus if available]

**📋 Quick Stats**
[5-6 bullet points: employees, sector, 52w range, dividend, analyst rating]

_Data sources: {sources}_

Rules:
- Use ONLY the data provided — never fabricate numbers
- If a metric is unavailable, skip it (don't say "N/A" repeatedly)  
- Keep total length under 600 words
- Telegram-friendly: use bold, bullets, emojis
- Be direct and useful — this is for a finance professional
"""


async def run_research(
    company_raw: str,
    user_profile: dict,
    db=None
) -> str:
    """
    Full research pipeline for a company.
    Returns formatted Telegram message string.
    """
    symbol  = resolve_ticker(company_raw)
    company = company_raw.title()

    logger.info(f"🔬 Starting research: {company} ({symbol})")

    # ── Step 1: Collect all data concurrently ─────────────────
    fundamentals, news, sec_data, web_results, peers = await asyncio.gather(
        _collect_fundamentals(symbol),
        _collect_news(symbol),
        _collect_sec(company),
        _collect_web(symbol, company),
        _collect_peers(symbol)
    )

    # ── Step 2: Build raw text corpus for BM25 ────────────────
    raw_parts = []

    # News text
    for n in news[:8]:
        headline = n.get("headline") or n.get("title") or ""
        summary  = n.get("summary") or n.get("snippet") or ""
        if headline:
            raw_parts.append(f"NEWS: {headline}. {summary}")

    # Web results text
    for w in web_results[:12]:
        title   = w.get("title") or ""
        snippet = w.get("snippet") or ""
        if snippet:
            raw_parts.append(f"WEB: {title}. {snippet}")

    # SEC data text
    if sec_data and not sec_data.get("error"):
        raw_parts.append(f"SEC: {json.dumps(sec_data)[:500]}")

    # Company description
    desc = fundamentals.get("description", "")
    if desc:
        raw_parts.append(f"DESCRIPTION: {desc}")

    raw_corpus = "\n\n".join(raw_parts)

    # ── Step 3: BM25 + Reranker over corpus ───────────────────
    rag_context = ""
    if raw_corpus:
        rag_context = await _rag_over_research(
            query=f"{company} business financials risks outlook competitors",
            raw_text=raw_corpus,
            top_k=8
        )

    # ── Step 4: Format news for prompt ────────────────────────
    news_formatted = ""
    for i, n in enumerate(news[:5], 1):
        headline = n.get("headline") or n.get("title") or ""
        date     = n.get("datetime") or n.get("date") or ""
        source   = n.get("source") or n.get("url") or ""
        if headline:
            date_str = f" ({date})" if date else ""
            news_formatted += f"{i}. {headline}{date_str} — {source}\n"

    if not news_formatted:
        news_formatted = "No recent news available."

    # ── Step 5: Build sources list ─────────────────────────────
    sources_used = []
    if fundamentals and not fundamentals.get("error"):
        sources_used.append(fundamentals.get("_source", "Yahoo Finance"))
    if news:
        sources_used.append("Finnhub News")
    if web_results:
        sources_used.append("Web Search")
    if sec_data and not sec_data.get("error"):
        sources_used.append("SEC EDGAR")
    sources_str = " · ".join(set(sources_used)) or "Multiple sources"

    # ── Step 6: 70b synthesis ─────────────────────────────────
    prompt = RESEARCH_PROMPT.format(
        company=f"{company} ({symbol})",
        user_profile=json.dumps(user_profile, ensure_ascii=False),
        fundamentals=json.dumps(
            {k: v for k, v in fundamentals.items()
             if v is not None and k not in ("description", "_source", "error")},
            ensure_ascii=False
        )[:1500],
        news=news_formatted,
        sec_info=json.dumps(sec_data, ensure_ascii=False)[:400] if sec_data else "Not found",
        rag_context=rag_context[:2500],
        peers=", ".join(peers) if peers else "Not available",
        sources=sources_str
    )

    try:
        response = await client.chat.completions.create(
            model=RESEARCH_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior financial research analyst. "
                        "Generate accurate, data-driven research reports. "
                        "Never fabricate numbers. If data is missing, skip that point."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=RESEARCH_TEMP
        )
        report = response.choices[0].message.content or "Research generation failed."

    except Exception as e:
        logger.error(f"Research LLM call failed: {e}")
        # Fallback: return structured data directly
        report = _fallback_report(company, symbol, fundamentals, news[:3], peers)

    return report


def _fallback_report(
    company: str,
    symbol: str,
    fundamentals: dict,
    news: list,
    peers: list
) -> str:
    """Structured fallback if LLM call fails — still useful output."""
    lines = [f"📊 *{company} ({symbol}) — Research Snapshot*\n"]

    # Financials
    if fundamentals and not fundamentals.get("error"):
        lines.append("*💰 Financials:*")
        if fundamentals.get("market_cap"):
            mc = fundamentals["market_cap"]
            mc_str = f"${mc/1e12:.2f}T" if mc > 1e12 else f"${mc/1e9:.1f}B"
            lines.append(f"  • Market Cap: {mc_str}")
        if fundamentals.get("revenue"):
            rev = fundamentals["revenue"]
            rev_str = f"${rev/1e12:.2f}T" if rev > 1e12 else f"${rev/1e9:.1f}B"
            lines.append(f"  • Revenue: {rev_str}")
        if fundamentals.get("pe_ratio"):
            lines.append(f"  • P/E Ratio: {fundamentals['pe_ratio']:.1f}")
        if fundamentals.get("profit_margin"):
            lines.append(f"  • Profit Margin: {fundamentals['profit_margin']*100:.1f}%")
        if fundamentals.get("analyst_rating"):
            lines.append(f"  • Analyst: {fundamentals['analyst_rating'].upper()}")

    # News
    if news:
        lines.append("\n*📰 Recent News:*")
        for n in news[:3]:
            h = n.get("headline") or n.get("title") or ""
            if h:
                lines.append(f"  • {h[:100]}")

    # Peers
    if peers:
        lines.append(f"\n*🏆 Peers:* {', '.join(peers[:4])}")

    lines.append("\n_Note: Full analysis temporarily unavailable. Data from Yahoo Finance / Finnhub._")
    return "\n".join(lines)
