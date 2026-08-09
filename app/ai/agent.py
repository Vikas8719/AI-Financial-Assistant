"""
Main AI Agent — Token-Optimized + Rate Limit Resilient
─────────────────────────────────────────────────────────────────
FIX 1: Groq embeddings removed (404 error) — memory.py handles this
FIX 2: Token limits reduced to save TPD quota:
        factual: 300 (was 512)
        analytical: 1200 (was 1800)
        max history: 6 turns (was 10)
FIX 3: Rate limit (429) error → user gets friendly message, not crash
FIX 4: Reranker model corrected — uses same GROQ_API_KEY
─────────────────────────────────────────────────────────────────
"""
import json
import re
from datetime import datetime, timezone
from typing import Any, Optional

from groq import AsyncGroq, RateLimitError
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

RERANKER_MODEL = "openai/gpt-oss-20b"
MAIN_MODEL     = "openai/gpt-oss-120b"

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
        r'\b52.week\b', r'\bvolume\b', r'\bopen\b', r'\bclose\b',
        r'\bhigh\b', r'\blow\b', r'\bchange\b', r'\bpercent\b',
        r'\b%\b', r'\bcrore\b', r'\bbillion\b', r'\bmillion\b',
        r'\byield\b', r'\brate\b', r'\bsubscriber[s]?\b', r'\barpu\b'
    ]
    if any(re.search(p, q) for p in factual):
        return "factual"

    compare = [
        r'\bcompare\b', r'\bvs\b', r'\bversus\b', r'\bbetter\b',
        r'\bwhich\b.*\bbetter\b', r'\bdifference\b', r'\bsimilar\b'
    ]
    if any(re.search(p, q) for p in compare):
        return "analytical"

    document = [
        r'\bdocument\b', r'\breport\b', r'\bfiling\b', r'\b10.?k\b',
        r'\b10.?q\b', r'\bannual\b', r'\bpdf\b', r'\bpage\b',
        r'\baccording to\b', r'\bstated\b', r'\bmentioned\b',
        r'\bsec\b', r'\bedgar\b', r'\baudit\b', r'\bnote\b'
    ]
    if any(re.search(p, q) for p in document):
        return "document"

    analytical = [
        r'\banalyze\b', r'\banalysis\b', r'\bwhy\b', r'\bhow\b',
        r'\bexplain\b', r'\bunderstand\b', r'\boutlook\b', r'\bforecast\b',
        r'\bguidance\b', r'\brisk\b', r'\bopportunity\b',
        r'\bcompetitor\b', r'\bindustry\b', r'\bsector\b',
        r'\btrend\b', r'\bperformance\b', r'\bgrowth\b',
        r'\bstrategy\b', r'\bvaluation\b', r'\bimpact\b'
    ]
    if any(re.search(p, q) for p in analytical):
        return "analytical"

    creative = [
        r'\bsummariz\b', r'\bbrief\b', r'\boverview\b', r'\bdigest\b',
        r'\btop news\b', r'\bmorning\b', r'\bwhat.s happening\b',
        r'\bupdate me\b', r'\bwhat happened\b'
    ]
    if any(re.search(p, q) for p in creative):
        return "creative"

    return "conversational"


# FIX 2: Reduced token limits to save TPD quota
TEMPERATURE_MAP = {
    "factual":        0.0,
    "document":       0.1,
    "analytical":     0.2,
    "creative":       0.4,
    "conversational": 0.4,
}

MAX_TOKENS_MAP = {
    "factual":        300,   # was 512  — saves 200 tokens/request
    "document":       900,   # was 1200
    "analytical":     1200,  # was 1800 — saves 600 tokens
    "creative":       600,   # was 900
    "conversational": 350,   # was 450
}


# ──────────────────────────────────────────────────────────────
#  Compare query parser
# ──────────────────────────────────────────────────────────────

def _extract_compare_companies(query: str) -> list[str]:
    q = query.lower()
    q = re.sub(
        r'\b(compare|comparison|between|versus|vs\.?|and|or|with|,)\b',
        ' ', q, flags=re.IGNORECASE
    )
    tokens = [t.strip() for t in q.split() if len(t.strip()) > 1]
    skip = {
        "the", "a", "an", "to", "for", "of", "me", "please", "show",
        "give", "tell", "their", "financials", "metrics", "stocks",
        "companies", "company", "stock", "price", "data"
    }
    candidates = [t for t in tokens if t not in skip]
    return candidates[:4] if candidates else []


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

        # ── Stage 0: Classify ─────────────────────────────────
        query_type  = classify_query(user_message)
        temperature = TEMPERATURE_MAP[query_type]
        max_tokens  = MAX_TOKENS_MAP[query_type]

        # ── Stage 1: Retrieval (concurrent) ───────────────────
        import asyncio

        async def _doc_retrieval():
            if not document_context or len(document_context.strip()) < 100:
                return None
            return self.rag.retrieve_candidates(
                query=user_message,
                document_content=document_context,
                top_k=15
            )

        async def _memory_retrieval():
            return await search_memory_hybrid(
                db=self.db,
                user_id=user_id,
                query=user_message,
                top_k=8
            )

        doc_result, hybrid_memories = await asyncio.gather(
            _doc_retrieval(),
            _memory_retrieval()
        )

        # ── Stage 2a: Rerank doc chunks ────────────────────────
        final_doc_context = ""
        doc_chunks_used   = 0

        if doc_result and doc_result.get("chunks"):
            reranked_chunks = await rerank_chunks(
                query=user_message,
                chunks=doc_result["chunks"],
                top_k=5,
                context_hint=f"User is a {user_profile.get('role', 'finance professional')}"
            )
            final_doc_context = self.rag.build_context_from_chunks(reranked_chunks)
            doc_chunks_used   = len(reranked_chunks)

            top_score = reranked_chunks[0].get("rerank_score", 0) if reranked_chunks else 0
            if top_score >= 6 and query_type not in ("factual",):
                query_type  = "document"
                temperature = TEMPERATURE_MAP["document"]
                max_tokens  = MAX_TOKENS_MAP["document"]

        # ── Stage 2b: Rerank memories ─────────────────────────
        rag_memory = ""
        if hybrid_memories:
            relevant_memories = await rerank_memory(
                query=user_message,
                memories=hybrid_memories,
                top_k=3
            )
            if relevant_memories:
                lines = []
                for i, mem in enumerate(relevant_memories, 1):
                    snippet = mem[:200] + "..." if len(mem) > 200 else mem  # was 250
                    lines.append(f"[Memory {i}]: {snippet}")
                rag_memory = "\n".join(lines)

        # ── Build system prompt ────────────────────────────────
        # FIX 2: Only last 6 turns (was 10) — saves ~400 tokens
        history = await get_recent_history(self.db, user_id, limit=6)
        now     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        user_profile_str = json.dumps(user_profile, ensure_ascii=False)

        context_parts = [f"Now: {now}"]
        if final_doc_context:
            context_parts.append(
                f"\n📄 DOCUMENT ({doc_chunks_used} sections):\n{final_doc_context[:3500]}"  # was 4500
            )
        if rag_memory:
            context_parts.append(f"\n🧠 MEMORY:\n{rag_memory}")

        system_content = SYSTEM_PROMPT.format(
            context      = "\n".join(context_parts),
            user_profile = user_profile_str,
            memory       = rag_memory or "None."
        )

        # ── Build messages ─────────────────────────────────────
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_content}
        ]
        for h in history[-6:]:  # FIX 2: 6 turns max
            role = h["role"]
            if role in ("user", "assistant", "system"):
                messages.append({"role": role, "content": h["content"]})  # type: ignore
        messages.append({"role": "user", "content": user_message})

        # ── LLM call with rate limit handling ─────────────────
        has_strong_context = bool(final_doc_context and doc_chunks_used >= 2)
        skip_tools = has_strong_context and query_type in ("document", "factual")

        try:
            if skip_tools:
                response = await main_client.chat.completions.create(
                    model       = MAIN_MODEL,
                    messages    = messages,
                    max_tokens  = max_tokens,
                    temperature = temperature
                )
            else:
                response = await main_client.chat.completions.create(
                    model       = MAIN_MODEL,
                    messages    = messages,
                    tools       = TOOLS,       # type: ignore
                    tool_choice = "auto",
                    max_tokens  = max_tokens,
                    temperature = temperature
                )
        except RateLimitError as e:
            # FIX 3: Friendly rate limit message instead of crash
            err_str = str(e)
            wait_match = re.search(r'try again in (\d+m\d+s|\d+s)', err_str)
            wait_time  = wait_match.group(1) if wait_match else "kuch minutes"
            await save_message(self.db, user_id, "user", user_message)
            return (
                f"⏳ Abhi bahut zyada requests aa rahi hain — {wait_time} mein dobara try karein.\n\n"
                f"_(Daily AI quota temporarily full — thodi der mein wapas aajao!)_"
            )
        except Exception as e:
            await save_message(self.db, user_id, "user", user_message)
            return "⚠️ AI response nahi mila. Please dobara try karein."

        msg = response.choices[0].message

        # ── Tool execution ─────────────────────────────────────
        if msg.tool_calls:
            tool_results = await self._execute_tools(
                msg.tool_calls, user_id, user_profile, user_message
            )

            assistant_msg: ChatCompletionMessageParam = {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [  # type: ignore
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in msg.tool_calls
                ]
            }
            messages.append(assistant_msg)

            for tool_call, result in zip(msg.tool_calls, tool_results):
                messages.append({
                    "role": "tool",          # type: ignore
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False)
                })

            try:
                final_response = await main_client.chat.completions.create(
                    model       = MAIN_MODEL,
                    messages    = messages,
                    max_tokens  = max_tokens,
                    temperature = min(temperature, 0.15)
                )
                answer = final_response.choices[0].message.content or "Response generate nahi hua."
            except RateLimitError:
                # FIX 3: Even tool synthesis can hit rate limit
                answer = "⏳ AI quota temporarily full. Thodi der mein dobara try karein."
            except Exception:
                answer = "⚠️ Response generate nahi hua. Please try again."
        else:
            answer = msg.content or "Response generate nahi hua."

        # ── Save to DB ─────────────────────────────────────────
        await save_message(self.db, user_id, "user", user_message)
        await save_message(self.db, user_id, "assistant", answer)

        return answer

    # ── Tool dispatcher ────────────────────────────────────────

    async def _execute_tools(
        self,
        tool_calls: Any,
        user_id: int,
        user_profile: dict,
        original_query: str = ""
    ) -> list:
        results = []
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            try:
                result = await self._call_tool(name, args, user_id, original_query)
            except Exception as e:
                result = {
                    "_tool_error": str(e),
                    "_tool": name,
                    "_note": "Data retrieval failed. Using available information."
                }
            results.append(result)
        return results

    async def _call_tool(
        self,
        name: str,
        args: dict,
        user_id: int,
        original_query: str = ""
    ) -> dict:

        if name == "get_stock_price":
            sym    = resolve_ticker(args.get("symbol", ""))
            result = await finnhub.get_quote(sym)
            if result.get("error"):
                result2 = await yahoo.get_fundamentals(sym)
                if not result2.get("error"):
                    return result2
                ws = await search.search(f"{sym} stock price today")
                return {**result, "_web_fallback": ws.get("answer", ""), "_source": "web"}
            return result

        elif name == "get_company_news":
            sym = resolve_ticker(args.get("symbol", ""))
            result = await finnhub.get_company_news(sym, args.get("days", 7))
            if result.get("error") or not result.get("news"):
                ws = await search.search(f"{sym} company news latest")
                return {"news": ws.get("results", []), "_source": "web_search"}
            return result

        elif name == "get_company_fundamentals":
            sym    = resolve_ticker(args.get("symbol", ""))
            result = await yahoo.get_fundamentals(sym)
            if result.get("error") or not _fundamentals_ok(result):
                from app.services.compare_service import _fetch_finnhub
                result2 = await _fetch_finnhub(sym)
                if not result2.get("error"):
                    return result2
                from app.services.compare_service import _fetch_web
                result3 = await _fetch_web(args.get("symbol", sym), sym)
                return result3
            return result

        elif name == "compare_companies":
            sym1_raw = args.get("symbol1", "")
            sym2_raw = args.get("symbol2", "")
            extra    = _extract_compare_companies(original_query)
            all_syms = list(dict.fromkeys([sym1_raw, sym2_raw] + extra))
            if len(all_syms) > 2:
                return await smart_compare_multi(all_syms[:4])
            else:
                return await smart_compare(sym1_raw, sym2_raw)

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
                return {"_web_fallback": ws.get("answer", ""), "results": ws.get("results", [])}
            return result

        elif name == "get_earnings_calendar":
            result = await finnhub.get_earnings_calendar(args.get("days", 7))
            if result.get("error"):
                ws = await search.search("earnings calendar this week companies reporting")
                return {"_web_fallback": ws.get("answer", ""), "_source": "web"}
            return result

        elif name == "set_alert":
            from app.models.user_repo import create_alert
            await create_alert(self.db, user_id, args)
            return {"status": "Alert created", "details": args}

        elif name == "get_gmail_summary":
            from app.services.google_service import GoogleService
            google = GoogleService(user_id, self.db)
            return await google.search_emails(args.get("query", ""))

        elif name == "get_calendar_events":
            from app.services.google_service import GoogleService
            google = GoogleService(user_id, self.db)
            return await google.get_upcoming_events(args.get("days", 7))

        elif name == "analyze_document":
            from app.services.document_service import DocumentService
            doc_service = DocumentService(self.db)
            return await doc_service.answer_question(
                user_id,
                args.get("question", ""),
                args.get("document_id")
            )

        else:
            return {"error": f"Unknown tool: {name}"}


def _fundamentals_ok(data: dict) -> bool:
    useful = ["market_cap", "revenue", "pe_ratio", "eps", "profit_margin"]
    return any(data.get(f) for f in useful)
