"""
REST API routes:
  POST /webhook      — Telegram webhook
  GET  /health       — Health check
  GET  /docs         — Auto-generated (debug only)
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
    """Liveness probe — confirms the service is up."""
    return {"status": "ok", "service": "FinBot", "version": "1.0.0"}


@router.post("/webhook", tags=["Telegram"])
async def telegram_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Telegram sends all bot updates here.
    Must return 200 quickly — heavy work is awaited inside handle_update.
    """
    try:
        payload = await request.json()
        await handle_update(payload, db)
        return {"ok": True}
    except Exception as e:
        logger.exception(f"Webhook error: {e}")
        # Always return 200 to Telegram so it doesn't retry indefinitely
        return JSONResponse(status_code=200, content={"ok": False, "error": str(e)})


@router.get("/me", tags=["Debug"])
async def bot_info():
    """Return bot identity — useful for confirming token is valid."""
    if not settings.DEBUG:
        raise HTTPException(status_code=404)
    from telegram import Bot
    b = Bot(token=settings.TELEGRAM_BOT_TOKEN)
    info = await b.get_me()
    return {
        "id": info.id,
        "username": info.username,
        "name": info.full_name,
    }
