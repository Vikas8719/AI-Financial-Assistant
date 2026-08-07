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

    id = Column(BigInteger, primary_key=True)
    username = Column(String(100), nullable=True)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    role = Column(String(100), nullable=True)
    watchlist = Column(JSONB, default=list)
    interests = Column(JSONB, default=list)
    briefing_time = Column(String(50), nullable=True)   # ← was VARCHAR(10), now 50
    timezone = Column(String(50), default="UTC")
    onboarded = Column(Boolean, default=False)
    google_tokens = Column(JSONB, nullable=True)
    preferences = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=True)
    condition = Column(String(50), nullable=True)
    threshold = Column(Float, nullable=True)
    description = Column(Text, nullable=True)
    active = Column(Boolean, default=True)
    last_triggered = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=True)
    summary = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Also alter existing column size in case table already exists
        await conn.execute(text(
            "ALTER TABLE users ALTER COLUMN briefing_time TYPE VARCHAR(50)"
            " USING briefing_time::VARCHAR(50)"
            " " # no-op if table doesn't exist yet — create_all handles it
        ) if False else text("SELECT 1"))  # skip alter, let create_all handle new schema
        await conn.run_sync(Base.metadata.create_all)
        # Patch existing column if it's still VARCHAR(10)
        try:
            await conn.execute(text(
                "ALTER TABLE users ALTER COLUMN briefing_time TYPE VARCHAR(50)"
            ))
        except Exception:
            pass  # Column might already be correct size
