from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, BigInteger, String, Text, Boolean, DateTime, Float, Integer, text
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from datetime import datetime
import ssl

from app.config import settings


def _build_engine():
    url = settings.DATABASE_URL
    if "sslmode" in url:
        url = url.split("?")[0]

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    return create_async_engine(
        url,
        echo=settings.DEBUG,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        connect_args={"ssl": ssl_ctx},
    )


engine = _build_engine()
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id            = Column(BigInteger, primary_key=True)
    username      = Column(String(100), nullable=True)
    first_name    = Column(String(100), nullable=True)
    last_name     = Column(String(100), nullable=True)
    role          = Column(String(100), nullable=True)
    watchlist     = Column(JSONB, default=list)
    interests     = Column(JSONB, default=list)
    briefing_time = Column(String(50),  nullable=True)
    timezone      = Column(String(50),  default="UTC")
    onboarded     = Column(Boolean,     default=False)
    google_tokens = Column(JSONB,       nullable=True)
    preferences   = Column(JSONB,       default=dict)
    created_at    = Column(DateTime,    default=datetime.utcnow)
    updated_at    = Column(DateTime,    default=datetime.utcnow, onupdate=datetime.utcnow)


class Conversation(Base):
    __tablename__ = "conversations"

    id         = Column(Integer,    primary_key=True, autoincrement=True)
    user_id    = Column(BigInteger, nullable=False, index=True)
    role       = Column(String(20), nullable=False)
    content    = Column(Text,       nullable=False)
    # Vector(1536) kept for backward compat with existing rows.
    # New rows from nomic-embed (768-dim) are stored padded OR
    # column is migrated to 768 via init_db migration below.
    embedding  = Column(Vector(1536), nullable=True)
    metadata_  = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)


class Alert(Base):
    __tablename__ = "alerts"

    id            = Column(Integer,    primary_key=True, autoincrement=True)
    user_id       = Column(BigInteger, nullable=False, index=True)
    alert_type    = Column(String(50), nullable=False)
    symbol        = Column(String(20), nullable=True)
    condition     = Column(String(50), nullable=True)
    threshold     = Column(Float,      nullable=True)
    description   = Column(Text,       nullable=True)
    active        = Column(Boolean,    default=True)
    last_triggered= Column(DateTime,   nullable=True)
    created_at    = Column(DateTime,   default=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"

    id         = Column(Integer,     primary_key=True, autoincrement=True)
    user_id    = Column(BigInteger,  nullable=False, index=True)
    filename   = Column(String(255), nullable=False)
    file_type  = Column(String(50),  nullable=True)
    summary    = Column(Text,        nullable=True)
    content    = Column(Text,        nullable=True)
    metadata_  = Column("metadata",  JSONB, default=dict)
    created_at = Column(DateTime,    default=datetime.utcnow)


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    async with engine.begin() as conn:
        # Ensure pgvector extension exists
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Create tables
        await conn.run_sync(Base.metadata.create_all)

        # ── Migration: resize embedding column 1536 → 768 ──────────
        # nomic-embed-text produces 768-dim vectors.
        # Old rows with 1536-dim embeddings become NULL (they'll be
        # re-embedded lazily on next query via BM25 fallback).
        try:
            await conn.execute(text("""
                DO $$
                BEGIN
                    -- Check current dimension
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'conversations'
                          AND column_name = 'embedding'
                    ) THEN
                        -- Null out old 1536-dim embeddings (incompatible with 768)
                        UPDATE conversations
                        SET embedding = NULL
                        WHERE embedding IS NOT NULL
                          AND vector_dims(embedding) != 768;

                        -- Alter column to 768-dim
                        ALTER TABLE conversations
                            ALTER COLUMN embedding TYPE vector(768)
                            USING embedding::vector(768);
                    END IF;
                END
                $$;
            """))
        except Exception:
            # Column may already be 768-dim — safe to ignore
            pass

        # ── Briefing_time column size fix ──────────────────────────
        try:
            await conn.execute(text(
                "ALTER TABLE users ALTER COLUMN briefing_time TYPE VARCHAR(50)"
            ))
        except Exception:
            pass

        # ── pgvector HNSW index for fast cosine search ─────────────
        try:
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_conversations_embedding_hnsw
                ON conversations
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """))
        except Exception:
            pass
