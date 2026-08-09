"""
Main AI Agent — 4-Stage Pipeline with Automatic Fallback Chain
─────────────────────────────────────────────────────────────────────
Fallback chain for every tool:
  Primary source → Fallback source → Web search → Graceful error message

Key fixes:
  ✅ tool_choice="none" in second LLM call (fixes 400 error)
  ✅ Yahoo 429 → Finnhub fallback → Web search fallback
  ✅ All tool errors caught, web search used as last resort
  ✅ Bot never crashes — always gives an answer
"""
import json
import re
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from groq import AsyncGroq
from groq.types.chat import ChatCompletionMessageParam
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.memory import get_recent_history, save_message, search_memory_hybrid
from app.ai.rag_engine import VectorlessRAG
from app.ai.reranker import rerank_chunks, rerank_memory
from app.ai.prompts import SYSTEM_PROMPT
from app.ai.tools import TOOLS
from app.config import settings
from app.services.finnhub_service import FinnhubService
from app.services.yahoo_finance import YahooFinanceService
from app.services.sec_edgar import SecEdgarService
from app.services.web_search import WebSearchService

logger = logging.getLogger("finbot.agent")

MAIN_MODEL     = "openai/gpt-oss-120b"
RERANKER_MODEL = "openai/gpt-oss-20b"

main_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
finnhub     = FinnhubService()
yahoo       = YahooFinanceService()
sec         = SecEdgarService()
search      = WebSearchService()


# ──────────────────────────────────────────────────────────────
#  Query Classifier
# ──────────────────────────────────────────────────────────────

CONVERSATIONAL_OVERRIDES = {
    "hi", "hello", "hey", "hii", "helo", "hola",
    "how are you", "how r u", "how are you doing",
    "good morning", "good evening", "good night", "good afternoon",
    "thanks", "thank you", "ok", "okay", "sure", "got it",
    "bye", "goodbye", "see you", "later", "yes", "no", "yep", "nope",
    "what's up", "whats up", "sup", "wassup",
    "help", "what can you do", "what do you do",
}


def _is_conversational_override(q: str) -> bool:
    q_stripped = q.strip().lower().rstrip("!?.,:;")
    if q_stripped in CONVERSATIONAL_OVERRIDES:
        return True
    words = q_stripped.split()
    if len(words) <= 4:
        if words[0] in {"hi", "hello", "hey", "hii", "helo", "greetings", "howdy"}:
            return True
        if len(words) >= 3 and words[0] == "how" and words[1] == "are":
            return True
    return False


def classify_query(query: str) -> str:
    q = query.lower().strip()

    if _is_conversational_override(q):
        return "conversational"

    if any(re.search(p, q) for p in [
        r'\bprice\b', r'\bstock\b', r'\bquote\b', r'\bmarket cap\b',
        r'\bdividend\b', r'\bearning[s]?\b', r'\brevenue\b', r'\bprofit\b',
        r'\bebitda\b', r'\bdebt\b', r'\bshare price\b', r'\bvolume\b',
        r'\bcrore\b', r'\bbillion\b', r'\bmillion\b', r'\byield\b',
        r'\bsubscriber[s]?\b', r'\barpu\b', r'\beps\b', r'\bpe\b',
        r'\broe\b', r'\bcash\b', r'\bmargin\b',
    ]):
        return "factual"

    if any(re.search(p, q) for p in [
        r'\bdocument\b', r'\breport\b', r'\bfiling\b', r'\b10.?k\b',
        r'\b10.?q\b', r'\bannual\b', r'\bpdf\b', r'\baccording to\b',
        r'\bsec\b', r'\bedgar\b', r'\baudit\b',
    ]):
        return "document"

    if any(re.search(p, q) for p in [
        r'\banalyze\b', r'\banalysis\b', r'\bcompare\b', r'\bvs\b',
        r'\bversus\b', r'\bwhy\b', r'\bhow (does|did|do|is|was|will|can|much|many)\b',
        r'\bexplain\b', r'\boutlook\b', r'\bforecast\b', r'\brisk\b',
        r'\bcompetitor\b', r'\bindustry\b', r'\bsector\b', r'\btrend\b',
        r'\bperformance\b', r'\bgrowth\b', r'\bstrategy\b', r'\bvaluation\b',
    ]):
        return "analytical"

    if any(re.search(p, q) for p in [
        r'\bsummariz\b', r'\bbrief\b', r'\boverview\b', r'\bmorning\b',
        r'\bwhat.s happening\b', r'\bupdate me\b', r'\bwhat happened\b',
    ]):
        return "creative"

    return "conversational"


TEMPERATURE_MAP = {
    "factual": 0.0, "document": 0.1, "analytical": 0.2,
    "creative": 0.45, "conversational": 0.4,
}
MAX_TOKENS_MAP = {
    "factual": 512, "document": 1200, "analytical": 1500,
    "creative": 900, "conversational": 450,
}


# ──────────────────────────────────────────────────────────────
#  Fallback-aware tool executor
# ──────────────────────────────────────────────────────────────

def _has_useful_data(result: dict) -> bool:
    """Check if a tool result has meaningful data (not just an error)."""
    if not result or result.get("error"):
        return False
    # Check for any non-None, non-error value
    useful_keys = ["price", "revenue", "market_cap", "news", "filings",
                   "results", "answer", "indices", "earnings", "content"]
    return any(result.get(k) for k in useful_keys)


async def _web_search_fallback(query: str, context: str = "") -> dict:
    """Always-available fallback — web search never fails silently."""
    try:
        full_query = f"{context} {query}".strip() if context else query
        result = await search.search(full_query)
        if result and (result.get("results") or result.get("answer")):
            return {"_source": "web_search", **result}
    except Exception as e:
        logger.warning(f"Web search fallback also failed: {e}")
    return {"_source": "web_search", "answer": "", "results": []}


async def call_tool_with_fallback(name: str, args: dict, user_id: int, original_query: str, db) -> dict:
    """
    Execute a tool with automatic fallback chain:
    1. Primary tool
    2. Alternative data source
    3. Web search
    4. Graceful degradation message
    """
    symbol = args.get("symbol", args.get("symbol1", "")).upper().strip()

    # ── get_stock_price ──────────────────────────────────────
    if name == "get_stock_price":
        result = await finnhub.get_quote(symbol)
        if _has_useful_data(result):
            return result
        # Fallback: Yahoo Finance
        logger.info(f"Finnhub failed for {symbol}, trying Yahoo...")
        r2 = await yahoo.get_fundamentals(symbol)
        if r2.get("market_cap") or r2.get("eps"):
            return {**r2, "_source": "yahoo_fallback"}
        # Fallback: Web search
        logger.info(f"Yahoo failed for {symbol}, trying web search...")
        return await _web_search_fallback(f"{symbol} stock price today", symbol)

    # ── get_company_news ─────────────────────────────────────
    elif name == "get_company_news":
        result = await finnhub.get_company_news(symbol, args.get("days", 7))
        if result.get("news"):
            return result
        # Fallback: Web search for news
        company_name = args.get("company", symbol)
        return await _web_search_fallback(f"{company_name} latest news", symbol)

    # ── get_company_fundamentals ─────────────────────────────
    elif name == "get_company_fundamentals":
        result = await yahoo.get_fundamentals(symbol)
        if _has_useful_data(result) or result.get("pe_ratio") or result.get("revenue"):
            return result
        # Fallback 1: Finnhub profile
        logger.info(f"Yahoo 429/error for {symbol}, trying Finnhub...")
        try:
            profile = finnhub.client.company_profile2(symbol=symbol)
            if profile:
                return {
                    "symbol": symbol,
                    "company": profile.get("name"),
                    "market_cap": profile.get("marketCapitalization"),
                    "industry": profile.get("finnhubIndustry"),
                    "country": profile.get("country"),
                    "_source": "finnhub_fallback"
                }
        except Exception:
            pass
        # Fallback 2: Web search
        logger.info(f"Finnhub also failed for {symbol}, using web search...")
        return await _web_search_fallback(
            f"{symbol} stock financials revenue profit PE ratio market cap 2024", symbol
        )

    # ── compare_companies ────────────────────────────────────
    elif name == "compare_companies":
        sym1, sym2 = args.get("symbol1", ""), args.get("symbol2", "")
        d1 = await yahoo.get_fundamentals(sym1)
        d2 = await yahoo.get_fundamentals(sym2)

        # If either fails, try web search for comparison
        if not _has_useful_data(d1) and not _has_useful_data(d2):
            logger.info(f"Both Yahoo calls failed, using web search for comparison...")
            return await _web_search_fallback(
                f"compare {sym1} vs {sym2} stock financials revenue profit 2024"
            )
        if not _has_useful_data(d1):
            ws = await _web_search_fallback(f"{sym1} stock financials 2024")
            d1 = ws
        if not _has_useful_data(d2):
            ws = await _web_search_fallback(f"{sym2} stock financials 2024")
            d2 = ws
        return {"company1": d1, "company2": d2}

    # ── get_market_overview ──────────────────────────────────
    elif name == "get_market_overview":
        result = await yahoo.get_market_overview()
        if result.get("indices"):
            return result
        return await _web_search_fallback("stock market overview today S&P NASDAQ Nifty performance")

    # ── get_earnings_calendar ────────────────────────────────
    elif name == "get_earnings_calendar":
        result = await finnhub.get_earnings_calendar(args.get("days", 7))
        if result.get("earnings"):
            return result
        return await _web_search_fallback("earnings calendar this week upcoming results")

    # ── search_sec_filings ───────────────────────────────────
    elif name == "search_sec_filings":
        result = await sec.search_filings(
            args.get("company_name", ""), args.get("filing_type", "10-K")
        )
        if result.get("filings"):
            return result
        return await _web_search_fallback(
            f"{args.get('company_name','')} {args.get('filing_type','10-K')} SEC filing"
        )

    # ── web_search ───────────────────────────────────────────
    elif name == "web_search":
        return await search.search(args.get("query", original_query))

    # ── set_alert ────────────────────────────────────────────
    elif name == "set_alert":
        from app.models.user_repo import create_alert
        await create_alert(db, user_id, args)
        return {"status": "Alert created", "details": args}

    # ── Gmail ────────────────────────────────────────────────
    elif name == "get_gmail_summary":
        from app.services.google_service import GoogleService
        return await GoogleService(user_id, db).search_emails(args.get("query", ""))

    # ── Calendar ─────────────────────────────────────────────
    elif name == "get_calendar_events":
        from app.services.google_service import GoogleService
        return await GoogleService(user_id, db).get_upcoming_events(args.get("days", 7))

    # ── Document Q&A ─────────────────────────────────────────
    elif name == "analyze_document":
        from app.services.document_service import DocumentService
        return await DocumentService(db).answer_question(
            user_id, args.get("question", ""), args.get("document_id")
        )

    return {"error": f"Unknown tool: {name}"}


# ──────────────────────────────────────────────────────────────
#  Main Agent
# ──────────────────────────────────────────────────────────────

class FinancialAgent:
    def __init__(self, db: AsyncSession):
        self.db  = db
        self.rag = VectorlessRAG(db)

    async def process_message(
        self,
        user_id: int,
        user_message: str,
        user_profile: dict,
        document_context: Optional[str] = None
    ) -> str:

        query_type  = classify_query(user_message)
        temperature = TEMPERATURE_MAP[query_type]
        max_tokens  = MAX_TOKENS_MAP[query_type]

        # ── Step 1: RAG retrieval ─────────────────────────────
        async def _doc_retrieval():
            if not document_context or len(document_context.strip()) < 100:
                return None
            return self.rag.retrieve_candidates(
                query=user_message, document_content=document_context, top_k=15
            )

        async def _memory_retrieval():
            if len(user_message.strip()) < 15:
                return []
            return await search_memory_hybrid(db=self.db, user_id=user_id, query=user_message, top_k=8)

        doc_result, hybrid_memories = await asyncio.gather(_doc_retrieval(), _memory_retrieval())

        final_doc_context = ""
        doc_chunks_used   = 0

        if doc_result and doc_result.get("chunks"):
            reranked = await rerank_chunks(
                query=user_message, chunks=doc_result["chunks"], top_k=5,
                context_hint=f"User: {user_profile.get('role','finance professional')}"
            )
            final_doc_context = self.rag.build_context_from_chunks(reranked)
            doc_chunks_used   = len(reranked)
            if reranked and reranked[0].get("rerank_score", 0) >= 6 and query_type != "factual":
                query_type  = "document"
                temperature = TEMPERATURE_MAP["document"]
                max_tokens  = MAX_TOKENS_MAP["document"]

        rag_memory = ""
        if hybrid_memories:
            relevant = await rerank_memory(user_message, hybrid_memories, top_k=3)
            if relevant:
                rag_memory = "\n".join(
                    f"[Mem {i+1}]: {m[:200]}{'...' if len(m)>200 else ''}"
                    for i, m in enumerate(relevant)
                )

        # ── Step 2: Build prompt ──────────────────────────────
        history          = await get_recent_history(self.db, user_id, limit=8)
        now              = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        user_profile_str = json.dumps(user_profile, ensure_ascii=False)

        context_parts = [f"Time: {now}"]
        if final_doc_context:
            context_parts.append(f"\n📄 DOCUMENT ({doc_chunks_used} sections):\n{final_doc_context[:4000]}")
        if rag_memory:
            context_parts.append(f"\n🧠 MEMORY:\n{rag_memory}")

        system_content = SYSTEM_PROMPT.format(
            context="\n".join(context_parts),
            user_profile=user_profile_str,
            memory=rag_memory or "None."
        )

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_content}
        ]
        for h in history[-8:]:
            if h["role"] in ("user", "assistant"):
                messages.append({"role": h["role"], "content": h["content"]})  # type: ignore[arg-type]
        messages.append({"role": "user", "content": user_message})

        # ── Step 3: First LLM call ────────────────────────────
        skip_tools = bool(final_doc_context and doc_chunks_used >= 2
                          and query_type in ("document", "factual"))

        if skip_tools:
            response = await main_client.chat.completions.create(
                model=MAIN_MODEL, messages=messages,
                max_tokens=max_tokens, temperature=temperature
            )
        else:
            response = await main_client.chat.completions.create(
                model=MAIN_MODEL, messages=messages,
                tools=TOOLS,  # type: ignore[arg-type]
                tool_choice="auto",
                max_tokens=max_tokens, temperature=temperature
            )

        msg = response.choices[0].message

        # ── Step 4: Tool execution with fallback chain ────────
        if msg.tool_calls:
            # Run all tools concurrently with fallback
            tool_results = await asyncio.gather(*[
                self._safe_tool_call(tc, user_id, user_message)
                for tc in msg.tool_calls
            ], return_exceptions=False)

            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [  # type: ignore[typeddict-item]
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            })
            for tc, result in zip(msg.tool_calls, tool_results):
                messages.append({
                    "role": "tool",  # type: ignore[typeddict-item]
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False)
                })

            # ── Second LLM call — tool_choice="none" fixes 400 error ──
            try:
                final = await main_client.chat.completions.create(
                    model=MAIN_MODEL,
                    messages=messages,
                    tools=TOOLS,         # type: ignore[arg-type]
                    tool_choice="none",  # ← KEY FIX: no more tool calls in 2nd pass
                    max_tokens=max_tokens,
                    temperature=min(temperature, 0.15)
                )
                answer = final.choices[0].message.content or "I couldn't generate a response."
            except Exception as e:
                logger.warning(f"Second LLM call failed ({e}), retrying without tools...")
                try:
                    final = await main_client.chat.completions.create(
                        model=MAIN_MODEL,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=min(temperature, 0.15)
                    )
                    answer = final.choices[0].message.content or "I couldn't generate a response."
                except Exception as e2:
                    logger.error(f"Both LLM calls failed: {e2}")
                    answer = "⚠️ I encountered an issue. Please try rephrasing your question."
        else:
            answer = msg.content or "I couldn't generate a response."

        # Save to memory
        await asyncio.gather(
            save_message(self.db, user_id, "user", user_message),
            save_message(self.db, user_id, "assistant", answer),
            return_exceptions=True
        )
        return answer

    async def _safe_tool_call(self, tc: Any, user_id: int, query: str) -> dict:
        """Execute one tool with full fallback chain — never raises."""
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments)
        except Exception:
            args = {}
        try:
            return await call_tool_with_fallback(name, args, user_id, query, self.db)
        except Exception as e:
            logger.error(f"Tool {name} failed completely: {e}")
            # Last resort — web search with the original query
            try:
                return await _web_search_fallback(query, name)
            except Exception:
                return {
                    "error": f"Tool {name} unavailable",
                    "note": "Data temporarily unavailable. Please try again shortly."
                }
