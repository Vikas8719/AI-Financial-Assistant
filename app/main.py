"""
FinBot — AI Financial Assistant
FastAPI entry point: webhook, startup, shutdown.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from telegram import Bot

from app.api.routes import router as api_router
from app.config import settings
from app.database import init_db
from app.scheduler.jobs import start_scheduler, shutdown_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("finbot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────
    logger.info("🚀 FinBot starting up...")
    await init_db()
    logger.info("✅ Database initialised")

    if settings.WEBHOOK_URL:
        try:
            bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
            webhook_url = f"{settings.WEBHOOK_URL.rstrip('/')}/webhook"
            await bot.set_webhook(webhook_url, drop_pending_updates=True)
            logger.info(f"✅ Telegram webhook set → {webhook_url}")
        except Exception as e:
            logger.warning(f"⚠️  Could not set webhook: {e}")

    start_scheduler()
    logger.info("✅ Scheduler started")

    yield  # App is running

    # ── Shutdown ─────────────────────────────────────────────
    logger.info("🛑 FinBot shutting down...")
    shutdown_scheduler()


app = FastAPI(
    title="FinBot — AI Financial Assistant",
    description="AI-powered financial assistant living inside Telegram.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/", include_in_schema=False)
async def root():
    return {"status": "FinBot is running 🤖", "version": "1.0.0"}
