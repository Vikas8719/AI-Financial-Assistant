"""Main AI Agent — orchestrates Groq LLM with tool calling."""
import json
from datetime import datetime, timezone
from typing import Any, Optional

from groq import AsyncGroq
from groq.types.chat import ChatCompletionMessageParam
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.memory import get_recent_history, get_conversation_summary, save_message
from app.ai.prompts import SYSTEM_PROMPT
from app.ai.tools import TOOLS
from app.config import settings
from app.services.finnhub_service import FinnhubService
from app.services.yahoo_finance import YahooFinanceService
from app.services.sec_edgar import SecEdgarService
from app.services.web_search import WebSearchService

client = AsyncGroq(api_key=settings.GROQ_API_KEY)
finnhub = FinnhubService()
yahoo = YahooFinanceService()
sec = SecEdgarService()
search = WebSearchService()


class FinancialAgent:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def process_message(
        self,
        user_id: int,
        user_message: str,
        user_profile: dict,
        document_context: Optional[str] = None
    ) -> str:
        history = await get_recent_history(self.db, user_id, limit=15)
        memory_summary = await get_conversation_summary(self.db, user_id)

        user_profile_str = json.dumps(user_profile, ensure_ascii=False)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        system_content = SYSTEM_PROMPT.format(
            context=f"Current time: {now}" + (
                f"\nDocument context: {document_context[:500]}" if document_context else ""
            ),
            user_profile=user_profile_str,
            memory=memory_summary
        )

        # Build typed message list
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_content}
        ]
        for h in history[-12:]:
            role = h["role"]
            if role in ("user", "assistant", "system"):
                messages.append({"role": role, "content": h["content"]})  # type: ignore[arg-type]
        messages.append({"role": "user", "content": user_message})

        # First LLM call
        response = await client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            tools=TOOLS,  # type: ignore[arg-type]
            tool_choice="auto",
            max_tokens=2048,
            temperature=0.3
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            tool_results = await self._execute_tools(msg.tool_calls, user_id, user_profile)

            # Append assistant turn with tool calls
            assistant_msg: ChatCompletionMessageParam = {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [  # type: ignore[typeddict-item]
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
                tool_msg: ChatCompletionMessageParam = {
                    "role": "tool",  # type: ignore[typeddict-item]
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                }
                messages.append(tool_msg)

            final_response = await client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=messages,
                max_tokens=1500,
                temperature=0.3
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
