"""
Alert monitor — checks price alerts and sends Telegram notifications.
Runs every 5 minutes via APScheduler.
"""
import logging
from datetime import datetime

from telegram import Bot

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.user_repo import get_all_price_alerts

logger = logging.getLogger("finbot.alerts")
bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)


def _format_price(value: float, currency: str = "USD") -> str:
    return f"{currency} {value:,.2f}" if value else "N/A"


def _build_alert_message(alert, quote: dict) -> str:
    symbol = alert.symbol or ""
    price = quote.get("price", 0)
    change_pct = quote.get("change_pct", 0)
    currency = quote.get("currency", "USD")
    company = quote.get("company", symbol)

    direction = "📈" if change_pct >= 0 else "📉"
    sign = "+" if change_pct >= 0 else ""

    return (
        f"🔔 *Alert triggered!*\n\n"
        f"*{company} ({symbol})*\n"
        f"Price: *{_format_price(price, currency)}*\n"
        f"Change: {direction} {sign}{change_pct:.2f}% today\n\n"
        f"_{alert.description}_"
    )


async def run_alert_monitor() -> None:
    """Check all active price alerts and notify users if conditions are met."""
    try:
        from app.services.finnhub_service import FinnhubService
        finnhub = FinnhubService()

        # Always create a fresh session inside the job
        async with AsyncSessionLocal() as db:
            alerts = await get_all_price_alerts(db)
            if not alerts:
                return

            logger.debug(f"Checking {len(alerts)} active alert(s)...")

            for alert in alerts:
                if not alert.symbol:
                    continue

                try:
                    quote = await finnhub.get_quote(alert.symbol)
                    if quote.get("error"):
                        continue

                    price = quote.get("price") or 0
                    change_pct = quote.get("change_pct") or 0
                    triggered = False

                    if alert.alert_type == "price_above" and alert.threshold and price >= alert.threshold:
                        triggered = True
                    elif alert.alert_type == "price_below" and alert.threshold and price <= alert.threshold:
                        triggered = True
                    elif alert.alert_type == "pct_change" and alert.threshold and abs(change_pct) >= alert.threshold:
                        triggered = True

                    if triggered:
                        message = _build_alert_message(alert, quote)
                        await bot.send_message(
                            chat_id=alert.user_id,
                            text=message,
                            parse_mode="Markdown"
                        )
                        alert.last_triggered = datetime.utcnow()
                        alert.active = False
                        db.add(alert)
                        logger.info(f"✅ Alert fired: user={alert.user_id} symbol={alert.symbol}")

                except Exception as e:
                    logger.warning(f"Alert check failed for {alert.symbol}: {e}")
                    continue

            await db.commit()

    except Exception as e:
        logger.error(f"Alert monitor crashed: {e}")
