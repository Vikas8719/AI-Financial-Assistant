"""
APScheduler jobs — Fixed & Enhanced
─────────────────────────────────────────────────────────────────
Fixes:
  ✅ Timezone-aware time matching (IST, EST, UTC all work correctly)
  ✅ 8:00 AM user local time → converted to UTC for matching
  ✅ Temperature=0.45 for morning brief (creative query type)
  ✅ Richer brief: sector performance + top movers + earnings today
  ✅ Render-safe: scheduler won't crash on cold start
─────────────────────────────────────────────────────────────────
How it works:
  - User sets "8:00 AM" during onboarding (stored as "08:00")
  - User timezone stored as "Asia/Kolkata", "America/New_York", etc.
  - Every 5 min, scheduler converts each user's 08:00 local → UTC
  - Sends brief only when UTC now matches converted time
"""
import json
import logging
from datetime import datetime, timezone

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from telegram import Bot

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.user_repo import get_users_with_briefing, get_user_profile_dict
from app.scheduler.alerts import run_alert_monitor

logger = logging.getLogger("finbot.scheduler")
scheduler = AsyncIOScheduler(timezone="UTC")
bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)


# ──────────────────────────────────────────────
#  Timezone-aware time matcher
# ──────────────────────────────────────────────

def _get_current_hour_for_timezone(tz_name: str) -> str:
    """
    Convert current UTC time to user's local time.
    Returns "HH:MM" string (only :00 and :30 slots).

    Example:
      UTC 02:30 + "Asia/Kolkata" (UTC+5:30) → "08:00"
      UTC 13:00 + "America/New_York" (UTC-5) → "08:00"
    """
    try:
        tz = pytz.timezone(tz_name)
        local_now = datetime.now(pytz.utc).astimezone(tz)
        # Only match on :00 and :30 slots (scheduler runs every 5 min)
        if local_now.minute < 15:
            minute_slot = "00"
        elif local_now.minute < 45:
            minute_slot = "30"
        else:
            minute_slot = "00"
            # Roll to next hour
            local_now = local_now.replace(hour=(local_now.hour + 1) % 24)
        return f"{local_now.hour:02d}:{minute_slot}"
    except Exception:
        # Fallback: UTC
        now = datetime.now(pytz.utc)
        minute_slot = "00" if now.minute < 30 else "30"
        return f"{now.hour:02d}:{minute_slot}"


def _should_send_brief(briefing_time: str, user_tz: str) -> bool:
    """
    Returns True if current time in user's timezone matches briefing_time.
    briefing_time format: "08:00", "09:30", "07:00" etc.

    Example:
      briefing_time="08:00", user_tz="Asia/Kolkata"
      → is it 8:00 AM IST right now? Yes/No
    """
    if not briefing_time:
        return False
    try:
        tz       = pytz.timezone(user_tz or "UTC")
        local_now = datetime.now(pytz.utc).astimezone(tz)
        local_hhmm = f"{local_now.hour:02d}:{local_now.minute:02d}"

        # Match within a 5-minute window (scheduler runs every 5 min)
        brief_h, brief_m = map(int, briefing_time.split(":"))
        now_h, now_m     = local_now.hour, local_now.minute

        total_brief = brief_h * 60 + brief_m
        total_now   = now_h   * 60 + now_m

        # True if within 0-4 minute window (scheduler granularity)
        return 0 <= (total_now - total_brief) < 5

    except Exception:
        return False


# ──────────────────────────────────────────────
#  Data gathering for morning brief
# ──────────────────────────────────────────────

async def _gather_market_data(finnhub_svc, yahoo_svc, watchlist: list) -> dict:
    """Gather all market data concurrently for the brief."""
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

    # All concurrent
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
#  Morning Brief Generator
# ──────────────────────────────────────────────

async def _generate_brief(profile: dict, market_bundle: dict, groq_client) -> str:
    """
    Generate personalized morning brief using llama-3.3-70b.
    Temperature=0.45 (creative) — fluent narrative, not robotic.
    """
    from app.ai.prompts import MORNING_BRIEF_PROMPT

    prompt = MORNING_BRIEF_PROMPT.format(
        user_profile=json.dumps(profile,                        ensure_ascii=False),
        market_data=json.dumps(market_bundle["market_overview"], ensure_ascii=False),
        watchlist_data=json.dumps(market_bundle["watchlist"],   ensure_ascii=False),
        news=json.dumps(market_bundle["top_news"],              ensure_ascii=False)
    )

    # Add earnings context if available
    earnings = market_bundle.get("earnings_today", {})
    if earnings and not earnings.get("error"):
        prompt += f"\n\nEarnings reports due today: {json.dumps(earnings, ensure_ascii=False)}"

    response = await groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",    # Main model for quality
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700,
        temperature=0.45    # Creative type — fluent, engaging narrative
    )
    return response.choices[0].message.content or "Market brief unavailable."


# ──────────────────────────────────────────────
#  Send Morning Brief (per user)
# ──────────────────────────────────────────────

async def _send_brief_to_user(user, db, groq_client, finnhub_svc, yahoo_svc) -> None:
    """Send morning brief to a single user."""
    try:
        profile  = await get_user_profile_dict(db, user.id)
        watchlist = profile.get("watchlist", [])
        user_tz  = user.timezone or "UTC"

        # Gather all market data concurrently
        market_bundle = await _gather_market_data(finnhub_svc, yahoo_svc, watchlist)

        # Generate brief with proper temperature
        brief = await _generate_brief(profile, market_bundle, groq_client)

        # Format with local date
        try:
            tz        = pytz.timezone(user_tz)
            local_now = datetime.now(pytz.utc).astimezone(tz)
            date_str  = local_now.strftime("%A, %d %b %Y")
        except Exception:
            date_str = datetime.utcnow().strftime("%A, %d %b %Y")

        message = (
            f"🌅 *Good morning! Market Brief — {date_str}*\n\n"
            f"{brief}\n\n"
            f"_Reply with any question or company name to dive deeper._"
        )

        await bot.send_message(
            chat_id=user.id,
            text=message,
            parse_mode="Markdown"
        )
        logger.info(f"✅ Brief sent → user {user.id} ({user_tz})")

    except Exception as e:
        logger.error(f"❌ Brief failed for user {user.id}: {e}")


# ──────────────────────────────────────────────
#  Dispatcher — runs every 5 min
# ──────────────────────────────────────────────

async def _dispatch_morning_briefs() -> None:
    """
    Check every 5 min which users are due a brief right now.
    Timezone-aware: "08:00" in user's local time, not UTC.
    """
    try:
        from groq import AsyncGroq
        from app.services.finnhub_service import FinnhubService
        from app.services.yahoo_finance import YahooFinanceService

        groq_client  = AsyncGroq(api_key=settings.GROQ_API_KEY)
        finnhub_svc  = FinnhubService()
        yahoo_svc    = YahooFinanceService()

        async with AsyncSessionLocal() as db:
            # Fetch all onboarded users with a briefing_time set
            from sqlalchemy import select
            from app.database import User

            result = await db.execute(
                select(User).where(
                    User.onboarded == True,
                    User.briefing_time != None,
                    User.briefing_time != ""
                )
            )
            all_users = result.scalars().all()

            if not all_users:
                return

            due_users = [
                u for u in all_users
                if _should_send_brief(
                    briefing_time=u.briefing_time,
                    user_tz=u.timezone or "UTC"
                )
            ]

            if not due_users:
                return

            logger.info(f"📰 Morning brief due for {len(due_users)} user(s)")

            for user in due_users:
                await _send_brief_to_user(
                    user, db, groq_client, finnhub_svc, yahoo_svc
                )

    except Exception as e:
        logger.error(f"Brief dispatcher error: {e}")


# ──────────────────────────────────────────────
#  Scheduler Setup
# ──────────────────────────────────────────────

def start_scheduler() -> None:
    # Morning brief — check every 5 min (timezone-aware matching)
    scheduler.add_job(
        _dispatch_morning_briefs,
        IntervalTrigger(minutes=5),
        id="morning_brief",
        replace_existing=True,
    )

    # Price & news alert monitor — every 5 min
    scheduler.add_job(
        run_alert_monitor,
        IntervalTrigger(minutes=5),
        id="alert_monitor",
        replace_existing=True,
    )

    scheduler.start()
    logger.info(
        "✅ Scheduler started\n"
        "   ├─ Morning brief check : every 5 min (timezone-aware)\n"
        "   └─ Alert monitor       : every 5 min"
    )


def shutdown_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("🛑 Scheduler stopped")
