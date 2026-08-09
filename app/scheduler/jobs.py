"""
APScheduler jobs — Timezone-aware Morning Brief + Alert Monitor
─────────────────────────────────────────────────────────────────
FIX: Bot() created lazily inside each job — not at module level.
     Module-level Bot() crashes on import if token is missing.
─────────────────────────────────────────────────────────────────
"""
import json
import logging
from datetime import datetime

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.user_repo import get_user_profile_dict
from app.scheduler.alerts import run_alert_monitor

logger    = logging.getLogger("finbot.scheduler")
scheduler = AsyncIOScheduler(timezone="UTC")


# ──────────────────────────────────────────────
#  Timezone helpers
# ──────────────────────────────────────────────

def _should_send_brief(briefing_time: str, user_tz: str) -> bool:
    """
    Returns True if current time in user's timezone matches briefing_time.
    Matches within a 5-minute window (scheduler granularity).
    """
    if not briefing_time:
        return False
    try:
        tz        = pytz.timezone(user_tz or "UTC")
        local_now = datetime.now(pytz.utc).astimezone(tz)

        brief_h, brief_m = map(int, briefing_time.split(":"))
        total_brief = brief_h * 60 + brief_m
        total_now   = local_now.hour * 60 + local_now.minute

        return 0 <= (total_now - total_brief) < 5

    except Exception:
        return False


# ──────────────────────────────────────────────
#  Market data gathering
# ──────────────────────────────────────────────

async def _gather_market_data(finnhub_svc, yahoo_svc, watchlist: list) -> dict:
    """Gather all market data concurrently."""
    import asyncio

    async def _get_watchlist_quotes():
        data = {}
        for symbol in watchlist[:6]:
            try:
                quote = await finnhub_svc.get_quote(symbol)
                if not quote.get("error"):
                    data[symbol] = quote
            except Exception:
                pass
        return data

    async def _get_market_overview():
        try:
            return await yahoo_svc.get_market_overview()
        except Exception:
            return {}

    async def _get_news():
        try:
            news_items = finnhub_svc.client.general_news("general", min_id=0)[:7]
            return [
                {
                    "headline": n.get("headline", ""),
                    "source":   n.get("source", ""),
                    "summary":  n.get("summary", "")[:150]
                }
                for n in news_items
                if n.get("headline")
            ]
        except Exception:
            return []

    async def _get_earnings_today():
        try:
            return await finnhub_svc.get_earnings_calendar(days=1)
        except Exception:
            return {}

    market_data, watchlist_data, news, earnings = await asyncio.gather(
        _get_market_overview(),
        _get_watchlist_quotes(),
        _get_news(),
        _get_earnings_today()
    )

    return {
        "market_overview": market_data,
        "watchlist":       watchlist_data,
        "top_news":        news,
        "earnings_today":  earnings
    }


# ──────────────────────────────────────────────
#  Brief generator
# ──────────────────────────────────────────────

async def _generate_brief(profile: dict, market_bundle: dict, groq_client) -> str:
    from app.ai.prompts import MORNING_BRIEF_PROMPT

    prompt = MORNING_BRIEF_PROMPT.format(
        user_profile   = json.dumps(profile,                         ensure_ascii=False),
        market_data    = json.dumps(market_bundle["market_overview"], ensure_ascii=False),
        watchlist_data = json.dumps(market_bundle["watchlist"],       ensure_ascii=False),
        news           = json.dumps(market_bundle["top_news"],        ensure_ascii=False)
    )

    earnings = market_bundle.get("earnings_today", {})
    if earnings and not earnings.get("error"):
        prompt += f"\n\nEarnings reports due today: {json.dumps(earnings, ensure_ascii=False)}"

    response = await groq_client.chat.completions.create(
        model       = "openai/gpt-oss-120b",
        messages    = [{"role": "user", "content": prompt}],
        max_tokens  = 700,
        temperature = 0.45
    )
    return response.choices[0].message.content or "Market brief unavailable."


# ──────────────────────────────────────────────
#  Per-user brief sender
# ──────────────────────────────────────────────

async def _send_brief_to_user(user, db, groq_client, finnhub_svc, yahoo_svc, bot) -> None:
    try:
        profile   = await get_user_profile_dict(db, user.id)
        watchlist = profile.get("watchlist", [])
        user_tz   = user.timezone or "UTC"

        market_bundle = await _gather_market_data(finnhub_svc, yahoo_svc, watchlist)
        brief         = await _generate_brief(profile, market_bundle, groq_client)

        try:
            tz        = pytz.timezone(user_tz)
            local_now = datetime.now(pytz.utc).astimezone(tz)
            date_str  = local_now.strftime("%A, %d %b %Y")
        except Exception:
            date_str = datetime.utcnow().strftime("%A, %d %b %Y")

        message = (
            f"🌅 *Good morning! Market Brief — {date_str}*\n\n"
            f"{brief}\n\n"
            f"_Koi sawaal poochhein ya company ka naam likhein._"
        )

        try:
            await bot.send_message(
                chat_id    = user.id,
                text       = message,
                parse_mode = "Markdown"
            )
        except Exception:
            # Markdown fail → plain text
            plain = message.replace("*", "").replace("_", "")
            await bot.send_message(chat_id=user.id, text=plain)

        logger.info(f"✅ Brief sent → user {user.id} ({user_tz})")

    except Exception as e:
        logger.error(f"❌ Brief failed for user {user.id}: {e}")


# ──────────────────────────────────────────────
#  Dispatcher — every 5 min
# ──────────────────────────────────────────────

async def _dispatch_morning_briefs() -> None:
    try:
        # FIX: All heavy imports inside the function — not at module level
        from groq import AsyncGroq
        from telegram import Bot
        from app.services.finnhub_service import FinnhubService
        from app.services.yahoo_finance import YahooFinanceService
        from sqlalchemy import select
        from app.database import User

        groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        finnhub_svc = FinnhubService()
        yahoo_svc   = YahooFinanceService()
        bot         = Bot(token=settings.TELEGRAM_BOT_TOKEN)

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(User).where(
                    User.onboarded     == True,
                    User.briefing_time != None,
                    User.briefing_time != ""
                )
            )
            all_users = result.scalars().all()
            if not all_users:
                return

            due_users = [
                u for u in all_users
                if _should_send_brief(u.briefing_time, u.timezone or "UTC")
            ]
            if not due_users:
                return

            logger.info(f"📰 Morning brief due for {len(due_users)} user(s)")
            for user in due_users:
                await _send_brief_to_user(
                    user, db, groq_client, finnhub_svc, yahoo_svc, bot
                )

    except Exception as e:
        logger.error(f"Brief dispatcher error: {e}")


# ──────────────────────────────────────────────
#  Scheduler setup
# ──────────────────────────────────────────────

def start_scheduler() -> None:
    scheduler.add_job(
        _dispatch_morning_briefs,
        IntervalTrigger(minutes=5),
        id              = "morning_brief",
        replace_existing = True,
    )
    scheduler.add_job(
        run_alert_monitor,
        IntervalTrigger(minutes=5),
        id              = "alert_monitor",
        replace_existing = True,
    )
    scheduler.start()
    logger.info(
        "✅ Scheduler started\n"
        "   ├─ Morning brief : every 5 min (timezone-aware)\n"
        "   └─ Alert monitor : every 5 min"
    )


def shutdown_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("🛑 Scheduler stopped")
