"""
APScheduler jobs:
  - Morning market brief (per user briefing_time)
  - Price & news alert monitor (every 5 min)
"""
import logging
from datetime import datetime

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from telegram import Bot

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.user_repo import get_users_with_briefing, get_user_profile_dict
from app.scheduler.alerts import run_alert_monitor

logger = logging.getLogger("finbot.scheduler")
scheduler = AsyncIOScheduler(timezone="UTC")
bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)


# ─── Morning Brief ────────────────────────────────────────────────────────────

async def _send_morning_briefs(hour_str: str) -> None:
    """Send morning brief to all users scheduled at this UTC hour."""
    async with AsyncSessionLocal() as db:
        users = await get_users_with_briefing(db, hour_str)
        if not users:
            return

        logger.info(f"📰 Sending morning brief to {len(users)} user(s) at {hour_str}")

        from app.ai.agent import FinancialAgent
        from app.services.finnhub_service import FinnhubService
        from app.services.yahoo_finance import YahooFinanceService
        from app.ai.prompts import MORNING_BRIEF_PROMPT
        from groq import AsyncGroq

        groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        finnhub_svc = FinnhubService()
        yahoo_svc = YahooFinanceService()

        for user in users:
            try:
                profile = await get_user_profile_dict(db, user.id)
                watchlist = profile.get("watchlist", [])

                # Gather market data
                market_data = await yahoo_svc.get_market_overview()
                watchlist_data = {}
                for symbol in watchlist[:6]:
                    quote = await finnhub_svc.get_quote(symbol)
                    if not quote.get("error"):
                        watchlist_data[symbol] = quote

                # Top 5 news items from Finnhub general market news
                try:
                    news_items = finnhub_svc.client.general_news("general", min_id=0)[:5]
                    news = [{"headline": n.get("headline"), "source": n.get("source")} for n in news_items]
                except Exception:
                    news = []

                import json
                prompt = MORNING_BRIEF_PROMPT.format(
                    user_profile=json.dumps(profile),
                    market_data=json.dumps(market_data),
                    watchlist_data=json.dumps(watchlist_data),
                    news=json.dumps(news)
                )

                response = await groq_client.chat.completions.create(
                    model=settings.GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=600,
                    temperature=0.3
                )
                brief = response.choices[0].message.content

                now_str = datetime.utcnow().strftime("%A, %d %b %Y")
                message = f"🌅 *Good morning! Here's your market brief for {now_str}*\n\n{brief}"
                await bot.send_message(chat_id=user.id, text=message, parse_mode="Markdown")
                logger.info(f"✅ Brief sent to user {user.id}")

            except Exception as e:
                logger.error(f"❌ Failed to send brief to user {user.id}: {e}")


async def _dispatch_morning_briefs() -> None:
    """Called every 30 min — checks which users are due a brief right now."""
    now_utc = datetime.utcnow()
    # Check exact hour and also :30 slot
    for minute in (0, 30):
        if now_utc.minute == minute:
            hour_str = f"{now_utc.hour:02d}:{minute:02d}"
            await _send_morning_briefs(hour_str)
            break


# ─── Scheduler Setup ──────────────────────────────────────────────────────────

def start_scheduler() -> None:
    # Morning brief check — every 30 minutes
    scheduler.add_job(
        _dispatch_morning_briefs,
        IntervalTrigger(minutes=30),
        id="morning_brief",
        replace_existing=True,
    )

    # Price & news alert monitor — every 5 minutes
    scheduler.add_job(
        run_alert_monitor,
        IntervalTrigger(minutes=5),
        id="alert_monitor",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("✅ Scheduler started (brief every 30 min | alerts every 5 min)")


def shutdown_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("🛑 Scheduler stopped")
