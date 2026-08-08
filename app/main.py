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
from app.database import init_db, AsyncSessionLocal
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

    # Setup PostgreSQL FTS indexes for vectorless RAG
    try:
        from app.ai.rag_engine import setup_fts_indexes
        async with AsyncSessionLocal() as db:
            await setup_fts_indexes(db)
        logger.info("✅ BM25 full-text search indexes ready")
    except Exception as e:
        logger.warning(f"⚠️  FTS index setup warning (non-fatal): {e}")

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
    logger.info("🤖 RAG Pipeline: BM25 → openai/gpt-oss-20b reranker → openai/gpt-oss-120b")

    yield

    # ── Shutdown ─────────────────────────────────────────────
    logger.info("🛑 FinBot shutting down...")
    shutdown_scheduler()


app = FastAPI(
    title="FinBot — AI Financial Assistant",
    description="AI-powered financial assistant living inside Telegram.",
    version="2.0.0",
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
    return {
        "status": "FinBot is running 🤖",
        "version": "2.0.0",
        "rag": "BM25 + LLM Reranker (openai/gpt-oss-20b)",
        "main_llm": "openai/gpt-oss-120b"
    }
