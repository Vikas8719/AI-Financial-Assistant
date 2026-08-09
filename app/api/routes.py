"""
REST API routes — Crash-proof webhook
  POST /webhook      — Telegram webhook (always returns 200)
  GET  /health       — Health check
  GET  /me           — Bot info (debug only)

FIX: Webhook never raises 500 — Telegram stops retrying on 500 errors,
     causing message loss. We always return 200 with error details in body.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import router as auth_router
from app.bot.telegram_handler import handle_update
from app.config import settings
from app.database import get_db

logger = logging.getLogger("finbot.routes")
router = APIRouter()
router.include_router(auth_router, prefix="/auth", tags=["OAuth"])


@router.get("/health", tags=["System"])
async def health_check():
    """Liveness probe."""
    return {"status": "ok", "service": "FinBot", "version": "2.0.0"}


@router.post("/webhook", tags=["Telegram"])
async def telegram_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Telegram sends all updates here.
    CRITICAL: Always return 200 — if we return 4xx/5xx, Telegram
    keeps retrying the same update for 24 hours and blocks new updates.
    """
    try:
        payload = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse webhook JSON: {e}")
        # Still 200 — bad payload shouldn't block future updates
        return JSONResponse(status_code=200, content={"ok": False, "error": "Invalid JSON"})

    try:
        await handle_update(payload, db)
        return JSONResponse(status_code=200, content={"ok": True})
    except Exception as e:
        logger.exception(f"Unhandled webhook error: {e}")
        # ALWAYS 200 to Telegram
        return JSONResponse(status_code=200, content={"ok": False, "error": str(e)})


@router.get("/me", tags=["Debug"])
async def bot_info():
    """Bot identity check — debug only."""
    if not settings.DEBUG:
        raise HTTPException(status_code=404)
    from telegram import Bot
    b    = Bot(token=settings.TELEGRAM_BOT_TOKEN)
    info = await b.get_me()
    return {
        "id":       info.id,
        "username": info.username,
        "name":     info.full_name,
    }
