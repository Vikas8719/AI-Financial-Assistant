"""
Main AI Agent — Speed-optimized 4-Stage Pipeline
─────────────────────────────────────────────────────────────────────
Speed tiers:
  FAST  (factual/conversational) → skip reranker → ~1.5s total
  NORMAL (analytical/creative)   → reranker runs → ~3s total
  DEEP  (document)               → full pipeline  → ~3.5s total

Key optimizations:
  ✅ Reranker skipped for factual+conversational (saves ~1s)
  ✅ Memory search skipped for very short queries (<15 chars)
  ✅ Tool calls + memory search run concurrently where possible
  ✅ max_tokens tightly capped per query type
  ✅ Temperature 0.0 for factual → Groq caches better
"""
import json
import re
import asyncio
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
from app.services.compare_service import smart_compare, smart_compare_multi, resolve_ticker

MAIN_MODEL  = "llama-3.3-70b-versatile"

main_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
finnhub     = FinnhubService()
yahoo       = YahooFinanceService()
sec         = SecEdgarService()
search      = WebSearchService()


# ──────────────────────────────────────────────────────────────
#  Query Classifier
# ──────────────────────────────────────────────────────────────

def classify_query(query: str) -> str:
    q = query.lower().strip()

    factual = [
        r'\bprice\b', r'\bstock\b', r'\bquote\b', r'\beps\b', r'\bpe\b',
        r'\bmarket cap\b', r'\bdividend\b', r'\bearning[s]?\b', r'\brevenue\b',
        r'\bprofit\b', r'\bebitda\b', r'\bdebt\b', r'\bshare price\b',
        r'\b52.week\b', r'\bvolume\b', r'\bhigh\b', r'\blow\b',
        r'\bchange\b', r'\bpercent\b', r'\b%\b', r'\bcrore\b',
        r'\bbillion\b', r'\bmillion\b', r'\byield\b', r'\brate\b',
        r'\bsubscriber[s]?\b', r'\barpu\b'
    ]
    if any(re.search(p, q) for p in factual):
        return "factual"

    compare = [r'\bcompare\b', r'\bvs\b', r'\bversus\b', r'\bbetter\b', r'\bdifference\b']
    if any(re.search(p, q) for p in compare):
        return "analytical"

    document = [
        r'\bdocument\b', r'\breport\b', r'\bfiling\b', r'\b10.?k\b',
        r'\b10.?q\b', r'\bannual\b', r'\bpdf\b', r'\baccording to\b',
        r'\bsec\b', r'\bedgar\b', r'\baudit\b'
    ]
    if any(re.search(p, q) for p in document):
        return "document"

    analytical = [
        r'\banalyze\b', r'\banalysis\b', r'\bwhy\b', r'\bhow\b',
        r'\bexplain\b', r'\boutlook\b', r'\bforecast\b', r'\brisk\b',
        r'\bcompetitor\b', r'\bindustry\b', r'\bsector\b', r'\btrend\b',
        r'\bperformance\b', r'\bgrowth\b', r'\bstrategy\b', r'\bvaluation\b'
    ]
    if any(re.search(p, q) for p in analytical):
        return "analytical"

    creative = [
        r'\bsummariz\b', r'\bbrief\b', r'\boverview\b', r'\bmorning\b',
        r'\bwhat.s happening\b', r'\bupdate me\b', r'\bwhat happened\b'
    ]
    if any(re.search(p, q) for p in creative):
        return "creative"

    return "conversational"


# Tight token caps — don't give LLM room to ramble
TEMPERATURE_MAP = {
    "factual":        0.0,
    "document":       0.1,
    "analytical":     0.2,
    "creative":       0.45,
    "conversational": 0.4,
}
MAX_TOKENS_MAP = {
    "factual":        400,   # Short precise answer
    "document":       1000,
    "analytical":     1200,
    "creative":       700,
    "conversational": 350,
}

# Query types that skip the reranker entirely (speed optimization)
SKIP_RERANKER_TYPES = {"factual", "conversational"}


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

        # ── Step 0: Classify ──────────────────────────────────
        query_type    = classify_query(user_message)
        temperature   = TEMPERATURE_MAP[query_type]
        max_tokens    = MAX_TOKENS_MAP[query_type]
        use_reranker  = query_type not in SKIP_RERANKER_TYPES
        short_query   = len(user_message.strip()) < 20  # e.g. "hi", "AAPL price"

        # ── Step 1: BM25 doc retrieval + memory (concurrent) ──
        async def _doc_retrieval():
            if not document_context or len(document_context.strip()) < 100:
                return None
            return self.rag.retrieve_candidates(
                query=user_message,
                document_content=document_context,
                top_k=15 if use_reranker else 5  # fewer candidates if no reranker
            )

        async def _memory_retrieval():
            # Skip memory for very short queries — not worth it
            if short_query:
                return []
            return await search_memory_hybrid(
                db=self.db,
                user_id=user_id,
                query=user_message,
                top_k=6 if use_reranker else 3
            )

        doc_result, hybrid_memories = await asyncio.gather(
            _doc_retrieval(),
            _memory_retrieval()
        )

        # ── Step 2: Rerank (only for analytical/document/creative) ──
        final_doc_context = ""
        doc_chunks_used   = 0
        rag_memory        = ""

        if doc_result and doc_result.get("chunks"):
            if use_reranker:
                reranked = await rerank_chunks(
                    query=user_message,
                    chunks=doc_result["chunks"],
                    top_k=5,
                    context_hint=f"finance pro: {user_profile.get('role','')}"
                )
            else:
                # BM25 order is fine for factual
                reranked = sorted(
                    doc_result["chunks"],
                    key=lambda c: c.get("bm25_score", 0),
                    reverse=True
                )[:5]
                for c in reranked:
                    c["reranked"] = False

            final_doc_context = self.rag.build_context_from_chunks(reranked)
            doc_chunks_used   = len(reranked)

            if use_reranker and reranked:
                top_score = reranked[0].get("rerank_score", 0)
                if top_score >= 6 and query_type != "factual":
                    query_type  = "document"
                    temperature = TEMPERATURE_MAP["document"]
                    max_tokens  = MAX_TOKENS_MAP["document"]

        if hybrid_memories:
            if use_reranker and len(hybrid_memories) > 2:
                relevant = await rerank_memory(user_message, hybrid_memories, top_k=3)
            else:
                relevant = hybrid_memories[:3]

            if relevant:
                lines = [
                    f"[Mem {i+1}]: {m[:200]}{'...' if len(m)>200 else ''}"
                    for i, m in enumerate(relevant)
                ]
                rag_memory = "\n".join(lines)

        # ── Step 3: Build prompt ───────────────────────────────
        history          = await get_recent_history(self.db, user_id, limit=8)
        now              = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        user_profile_str = json.dumps(user_profile, ensure_ascii=False)

        context_parts = [f"Time: {now}"]
        if final_doc_context:
            context_parts.append(
                f"\n📄 DOCUMENT ({doc_chunks_used} sections):\n{final_doc_context[:4000]}"
            )
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
        for h in history[-6:]:     # 6 turns, not 8 — saves tokens
            if h["role"] in ("user", "assistant"):
                messages.append({"role": h["role"], "content": h["content"]})  # type: ignore[arg-type]
        messages.append({"role": "user", "content": user_message})

        # ── Step 4: Main LLM ───────────────────────────────────
        skip_tools = bool(final_doc_context and doc_chunks_used >= 2
                          and query_type in ("document", "factual"))

        if skip_tools:
            response = await main_client.chat.completions.create(
                model=MAIN_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        else:
            response = await main_client.chat.completions.create(
                model=MAIN_MODEL,
                messages=messages,
                tools=TOOLS,  # type: ignore[arg-type]
                tool_choice="auto",
                max_tokens=max_tokens,
                temperature=temperature
            )

        msg = response.choices[0].message

        # ── Tool execution ─────────────────────────────────────
        if msg.tool_calls:
            tool_results = await self._execute_tools(
                msg.tool_calls, user_id, user_profile, user_message
            )

            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [  # type: ignore[typeddict-item]
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    }
                    for tc in msg.tool_calls
                ]
            })
            for tc, result in zip(msg.tool_calls, tool_results):
                messages.append({
                    "role": "tool",  # type: ignore[typeddict-item]
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False)
                })

            final = await main_client.chat.completions.create(
                model=MAIN_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=min(temperature, 0.15)
            )
            answer = final.choices[0].message.content or "I couldn't generate a response."
        else:
            answer = msg.content or "I couldn't generate a response."

        # ── Save (fire-and-forget style — don't await both) ────
        await asyncio.gather(
            save_message(self.db, user_id, "user",      user_message),
            save_message(self.db, user_id, "assistant", answer),
            return_exceptions=True
        )

        return answer

    async def _execute_tools(
        self, tool_calls: Any, user_id: int, user_profile: dict, query: str = ""
    ) -> list:
        # Run all tool calls concurrently
        tasks = []
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            tasks.append(self._call_tool_safe(name, args, user_id, query))
        return list(await asyncio.gather(*tasks))

    async def _call_tool_safe(self, name: str, args: dict, user_id: int, query: str) -> dict:
        try:
            return await self._call_tool(name, args, user_id, query)
        except Exception as e:
            return {"_note": "Data temporarily unavailable.", "_error": str(e), "_tool": name}

    async def _call_tool(self, name: str, args: dict, user_id: int, query: str = "") -> dict:
        if name == "get_stock_price":
            sym    = resolve_ticker(args.get("symbol", ""))
            result = await finnhub.get_quote(sym)
            if result.get("error"):
                r2 = await yahoo.get_fundamentals(sym)
                return r2 if not r2.get("error") else result
            return result

        elif name == "get_company_news":
            sym    = resolve_ticker(args.get("symbol", ""))
            result = await finnhub.get_company_news(sym, args.get("days", 7))
            if result.get("error") or not result.get("news"):
                ws = await search.search(f"{sym} company news latest")
                return {"news": ws.get("results", []), "_source": "web"}
            return result

        elif name == "get_company_fundamentals":
            sym    = resolve_ticker(args.get("symbol", ""))
            result = await yahoo.get_fundamentals(sym)
            if not _fundamentals_ok(result):
                from app.services.compare_service import _fetch_finnhub, _fetch_web
                r2 = await _fetch_finnhub(sym)
                if not r2.get("error"):
                    return r2
                return await _fetch_web(args.get("symbol", sym), sym)
            return result

        elif name == "compare_companies":
            sym1, sym2 = args.get("symbol1", ""), args.get("symbol2", "")
            extra      = _extract_compare_companies(query)
            all_syms   = list(dict.fromkeys([sym1, sym2] + extra))
            if len(all_syms) > 2:
                return await smart_compare_multi(all_syms[:4])
            return await smart_compare(sym1, sym2)

        elif name == "search_sec_filings":
            return await sec.search_filings(
                args.get("company_name", ""), args.get("filing_type", "10-K")
            )

        elif name == "web_search":
            return await search.search(args.get("query", ""))

        elif name == "get_market_overview":
            result = await yahoo.get_market_overview()
            if not result or result.get("error"):
                ws = await search.search("stock market overview today S&P NASDAQ")
                return {"_web": ws.get("answer", ""), "results": ws.get("results", [])}
            return result

        elif name == "get_earnings_calendar":
            result = await finnhub.get_earnings_calendar(args.get("days", 7))
            if result.get("error"):
                ws = await search.search("earnings calendar this week")
                return {"_web": ws.get("answer", ""), "_source": "web"}
            return result

        elif name == "set_alert":
            from app.models.user_repo import create_alert
            await create_alert(self.db, user_id, args)
            return {"status": "Alert created", "details": args}

        elif name == "get_gmail_summary":
            from app.services.google_service import GoogleService
            return await GoogleService(user_id, self.db).search_emails(args.get("query", ""))

        elif name == "get_calendar_events":
            from app.services.google_service import GoogleService
            return await GoogleService(user_id, self.db).get_upcoming_events(args.get("days", 7))

        elif name == "analyze_document":
            from app.services.document_service import DocumentService
            return await DocumentService(self.db).answer_question(
                user_id, args.get("question", ""), args.get("document_id")
            )

        return {"error": f"Unknown tool: {name}"}


def _fundamentals_ok(data: dict) -> bool:
    return any(data.get(f) for f in ["market_cap", "revenue", "pe_ratio", "eps", "profit_margin"])


def _extract_compare_companies(query: str) -> list[str]:
    q = re.sub(
        r'\b(compare|comparison|between|versus|vs\.?|and|or|with|,)\b',
        ' ', query, flags=re.IGNORECASE
    )
    tokens = [t.strip() for t in q.split() if len(t.strip()) > 1]
    skip   = {"the","a","an","to","for","of","me","please","show","give",
               "tell","their","financials","metrics","stocks","companies","company"}
    return [t for t in tokens if t.lower() not in skip][:4]
