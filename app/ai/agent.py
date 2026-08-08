"""
Main AI Agent — 4-Stage Pipeline
"""
import json
import re
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

RERANKER_MODEL = "openai/gpt-oss-20b"
MAIN_MODEL     = "openai/gpt-oss-120b"

main_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
finnhub     = FinnhubService()
yahoo       = YahooFinanceService()
sec         = SecEdgarService()
search      = WebSearchService()


# ──────────────────────────────────────────────────────────────
#  Query Classifier — Fixed
# ──────────────────────────────────────────────────────────────

# Short conversational messages that should NEVER be analytical
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
    """Check if query is purely conversational — before any pattern matching."""
    q_stripped = q.strip().lower().rstrip("!?.,:;")

    # Direct match
    if q_stripped in CONVERSATIONAL_OVERRIDES:
        return True

    # Short greeting patterns (under 5 words)
    words = q_stripped.split()
    if len(words) <= 4:
        greeting_words = {"hi", "hello", "hey", "hii", "helo", "greetings", "howdy"}
        if words[0] in greeting_words:
            return True
        # "how are you *" with <= 4 words
        if len(words) >= 3 and words[0] == "how" and words[1] == "are":
            return True

    return False


def _sanitize_ts_query(tokens: list[str]) -> str:
    """
    ✅ FIX for Bug 2: Sanitize tokens before building PostgreSQL ts_query.

    Removes/escapes characters that crash PostgreSQL FTS:
    - & (AND operator)
    - | (OR operator)
    - ! (NOT operator)
    - ( ) (grouping)
    - : (lexeme weight)
    - * (prefix match — only valid at end)
    - ? (not valid in tsquery)
    - ' (quote)
    - " (double quote)
    - Numbers-only tokens (no lexeme value)
    - Empty strings

    Returns a safe OR-joined ts_query string.
    """
    INVALID_CHARS = re.compile(r"[&|!():*?'\"\\/]")
    safe_tokens = []

    for token in tokens:
        # Strip invalid chars
        cleaned = INVALID_CHARS.sub("", token).strip()

        # Skip empty, numbers-only, or too-short tokens
        if not cleaned or len(cleaned) < 2:
            continue
        if cleaned.isdigit():
            continue

        # Must have at least one alphabetic character
        if not any(c.isalpha() for c in cleaned):
            continue

        safe_tokens.append(cleaned)

    if not safe_tokens:
        return ""

    # Use OR (|) joining — more permissive, better recall
    return " | ".join(safe_tokens[:8])


def classify_query(query: str) -> str:
    """
    ✅ FIX for Bug 1: Check conversational overrides FIRST.

    Problem was: 'how' in analytical patterns caused
    'Hello how are you' → 'analytical' (WRONG)

    Fix: short greetings and casual phrases are intercepted
    before any regex pattern matching runs.
    """
    q = query.lower().strip()

    # ── STEP 1: Conversational override (must come FIRST) ──────
    if _is_conversational_override(q):
        return "conversational"

    # ── STEP 2: Factual — specific data points ──────────────────
    factual_patterns = [
        r'\bprice\b', r'\bstock\b', r'\bquote\b', r'\beps\b', r'\bpe\b',
        r'\bmarket cap\b', r'\bdividend\b', r'\bearning[s]?\b', r'\brevenue\b',
        r'\bprofit\b', r'\bebitda\b', r'\bdebt\b', r'\bshare price\b',
        r'\b52.week\b', r'\bvolume\b', r'\bcrore\b', r'\bbillion\b',
        r'\bmillion\b', r'\byield\b', r'\bsubscriber[s]?\b', r'\barpu\b',
        r'\beps\b', r'\broe\b', r'\broa\b', r'\bcash\b', r'\bmargin\b',
    ]
    if any(re.search(p, q) for p in factual_patterns):
        return "factual"

    # ── STEP 3: Document — report/filing queries ─────────────────
    document_patterns = [
        r'\bdocument\b', r'\breport\b', r'\bfiling\b', r'\b10.?k\b',
        r'\b10.?q\b', r'\bannual\b', r'\bpdf\b', r'\bpage\b',
        r'\baccording to\b', r'\bstated\b', r'\bmentioned\b',
        r'\bsec\b', r'\bedgar\b', r'\baudit\b',
    ]
    if any(re.search(p, q) for p in document_patterns):
        return "document"

    # ── STEP 4: Analytical — compare/analyze/explain ─────────────
    # NOTE: 'how' removed — causes false positives on "how are you"
    # Use more specific multi-word patterns for how-based queries
    analytical_patterns = [
        r'\banalyze\b', r'\banalysis\b', r'\bcompare\b', r'\bvs\b',
        r'\bversus\b', r'\bwhy\b', r'\bhow (does|did|do|is|was|will|can|much|many)\b',
        r'\bexplain\b', r'\bunderstand\b', r'\boutlook\b', r'\bforecast\b',
        r'\bguidance\b', r'\brisk\b', r'\bopportunity\b',
        r'\bcompetitor\b', r'\bindustry\b', r'\bsector\b',
        r'\btrend\b', r'\bperformance\b', r'\bgrowth\b',
        r'\bstrategy\b', r'\bvaluation\b', r'\bimpact\b',
    ]
    if any(re.search(p, q) for p in analytical_patterns):
        return "analytical"

    # ── STEP 5: Creative — summaries, news briefs ────────────────
    creative_patterns = [
        r'\bsummariz\b', r'\bbrief\b', r'\boverview\b', r'\bdigest\b',
        r'\btop news\b', r'\bmorning\b', r'\bwhat.s happening\b',
        r'\bupdate me\b', r'\bwhat happened\b',
    ]
    if any(re.search(p, q) for p in creative_patterns):
        return "creative"

    return "conversational"


TEMPERATURE_MAP = {
    "factual":        0.0,
    "document":       0.1,
    "analytical":     0.2,
    "creative":       0.45,
    "conversational": 0.4,
}

MAX_TOKENS_MAP = {
    "factual":        512,
    "document":       1200,
    "analytical":     1500,
    "creative":       900,
    "conversational": 450,
}


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
                    snippet = mem[:250] + "..." if len(mem) > 250 else mem
                    lines.append(f"[Memory {i}]: {snippet}")
                rag_memory = "\n".join(lines)

        history = await get_recent_history(self.db, user_id, limit=10)
        now     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        user_profile_str = json.dumps(user_profile, ensure_ascii=False)

        context_parts = [f"Current time: {now}"]

        if final_doc_context:
            context_parts.append(
                f"\n📄 DOCUMENT EXCERPTS ({doc_chunks_used} sections):\n\n{final_doc_context[:4500]}"
            )

        if rag_memory:
            context_parts.append(f"\n🧠 RELEVANT PAST CONTEXT:\n{rag_memory}")

        system_content = SYSTEM_PROMPT.format(
            context="\n".join(context_parts),
            user_profile=user_profile_str,
            memory=rag_memory or "No relevant past memory."
        )

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_content}
        ]
        for h in history[-8:]:
            role = h["role"]
            if role in ("user", "assistant", "system"):
                messages.append({"role": role, "content": h["content"]})  # type: ignore[arg-type]
        messages.append({"role": "user", "content": user_message})

        has_strong_context = bool(final_doc_context and doc_chunks_used >= 2)
        skip_tools = has_strong_context and query_type in ("document", "factual")

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

        if msg.tool_calls:
            tool_results = await self._execute_tools(msg.tool_calls, user_id, user_profile)

            assistant_msg: ChatCompletionMessageParam = {
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
            }
            messages.append(assistant_msg)

            for tool_call, result in zip(msg.tool_calls, tool_results):
                messages.append({
                    "role": "tool",          # type: ignore[typeddict-item]
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

            final_response = await main_client.chat.completions.create(
                model=MAIN_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=min(temperature, 0.15)
            )
            answer = final_response.choices[0].message.content or "I couldn't generate a response."
        else:
            answer = msg.content or "I couldn't generate a response."

        await save_message(self.db, user_id, "user", user_message)
        await save_message(self.db, user_id, "assistant", answer)

        return answer

    async def _execute_tools(self, tool_calls: Any, user_id: int, user_profile: dict) -> list:
        results = []
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            try:
                result = await self._call_tool(name, args, user_id)
            except Exception as e:
                result = {"error": str(e), "tool": name}
            results.append(result)
        return results

    async def _call_tool(self, name: str, args: dict, user_id: int) -> dict:
        if name == "get_stock_price":
            return await finnhub.get_quote(args["symbol"])
        elif name == "get_company_news":
            return await finnhub.get_company_news(args["symbol"], args.get("days", 7))
        elif name == "get_company_fundamentals":
            return await yahoo.get_fundamentals(args["symbol"])
        elif name == "search_sec_filings":
            return await sec.search_filings(args["company_name"], args.get("filing_type", "10-K"))
        elif name == "web_search":
            return await search.search(args["query"])
        elif name == "get_market_overview":
            return await yahoo.get_market_overview()
        elif name == "get_earnings_calendar":
            return await finnhub.get_earnings_calendar(args.get("days", 7))
        elif name == "compare_companies":
            d1 = await yahoo.get_fundamentals(args["symbol1"])
            d2 = await yahoo.get_fundamentals(args["symbol2"])
            return {"company1": d1, "company2": d2}
        elif name == "set_alert":
            from app.models.user_repo import create_alert
            await create_alert(self.db, user_id, args)
            return {"status": "Alert created", "details": args}
        elif name == "get_gmail_summary":
            from app.services.google_service import GoogleService
            google = GoogleService(user_id, self.db)
            return await google.search_emails(args["query"])
        elif name == "get_calendar_events":
            from app.services.google_service import GoogleService
            google = GoogleService(user_id, self.db)
            return await google.get_upcoming_events(args.get("days", 7))
        elif name == "analyze_document":
            from app.services.document_service import DocumentService
            doc_service = DocumentService(self.db)
            return await doc_service.answer_question(
                user_id, args.get("question", ""), args.get("document_id")
            )
        else:
            return {"error": f"Unknown tool: {name}"}
