"""
Hybrid Memory — BM25 (keyword) + pgvector (semantic, optional)
─────────────────────────────────────────────────────────────────
FIX 1: Groq does NOT support /embeddings endpoint → 404 error.
        Embeddings silently disabled — BM25 handles retrieval alone.
        When embeddings become available (OpenAI key added), it auto-enables.

FIX 2: Token usage reduced — memory search no longer calls LLM.
        BM25 is free (PostgreSQL), no API calls, no rate limits.
─────────────────────────────────────────────────────────────────
"""
from datetime import datetime
from typing import List, Optional

from sqlalchemy import select, desc, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Conversation
from app.config import settings


def _now() -> datetime:
    return datetime.utcnow()


# ──────────────────────────────────────────────
#  Embedding (Optional — disabled if no OpenAI key)
# ──────────────────────────────────────────────

async def get_embedding(text_input: str) -> Optional[List[float]]:
    """
    FIX: Groq does NOT have /embeddings endpoint (404).
    Only attempt if OPENAI_API_KEY is set.
    Falls back to None silently — BM25 still works without embeddings.
    """
    if not text_input or len(text_input.strip()) < 3:
        return None

    # Check if OpenAI key is available (optional feature)
    openai_key = getattr(settings, "OPENAI_API_KEY", None)
    if not openai_key:
        return None   # BM25-only mode — no embeddings, no problem

    try:
        import httpx
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "text-embedding-3-small",  # Cheapest OpenAI embedding
                    "input": text_input[:512]
                }
            )
            if response.status_code == 200:
                data = response.json()
                return data["data"][0]["embedding"]
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────
#  Save Message
# ──────────────────────────────────────────────

async def save_message(
    db: AsyncSession,
    user_id: int,
    role: str,
    content: str,
    metadata: Optional[dict] = None
) -> Conversation:
    """
    Save message. Embedding only attempted if OpenAI key available.
    BM25 search works perfectly without embeddings.
    """
    embedding = None
    if role == "user":
        embedding = await get_embedding(content)   # Returns None if no OpenAI key

    msg = Conversation(
        user_id   = user_id,
        role      = role,
        content   = content,
        embedding = embedding,
        metadata_ = metadata or {},
        created_at = _now()
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg


# ──────────────────────────────────────────────
#  pgvector Semantic Search (only if embeddings available)
# ──────────────────────────────────────────────

async def search_memory_semantic(
    db: AsyncSession,
    user_id: int,
    query: str,
    limit: int = 10
) -> List[str]:
    """Semantic search — skipped if no OpenAI key (returns [])."""
    query_embedding = await get_embedding(query)
    if not query_embedding:
        return []   # BM25 handles everything

    try:
        vector_str = f"[{','.join(map(str, query_embedding))}]"
        result = await db.execute(
            text("""
                SELECT content,
                       1 - (embedding <=> :embedding::vector) AS similarity
                FROM conversations
                WHERE user_id   = :user_id
                  AND role      = 'user'
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> :embedding::vector
                LIMIT :limit
            """),
            {"user_id": user_id, "embedding": vector_str, "limit": limit}
        )
        rows = result.fetchall()
        return [row[0] for row in rows if row[1] > 0.3]
    except Exception:
        return []


# ──────────────────────────────────────────────
#  BM25 Keyword Search — PRIMARY retrieval method
# ──────────────────────────────────────────────

async def search_memory_bm25(
    db: AsyncSession,
    user_id: int,
    query: str,
    limit: int = 10
) -> List[str]:
    """
    PostgreSQL full-text search — free, no API calls, no rate limits.
    Catches: tickers (AAPL), numbers, exact terms (EBITDA, PE ratio).
    """
    try:
        from app.ai.rag_engine import tokenize_financial
        query_tokens = tokenize_financial(query)
    except Exception:
        query_tokens = [w for w in query.split() if len(w) > 2]

    if not query_tokens:
        # Fallback: recent history
        result = await db.execute(
            text("""
                SELECT content FROM conversations
                WHERE user_id = :user_id AND role = 'user'
                ORDER BY created_at DESC
                LIMIT :limit
            """),
            {"user_id": user_id, "limit": limit}
        )
        return [row[0] for row in result.fetchall()]

    try:
        ts_query = " | ".join(query_tokens[:8])
        result = await db.execute(
            text("""
                SELECT content,
                       ts_rank(
                           to_tsvector('english', content),
                           to_tsquery('english', :tsquery)
                       ) AS rank
                FROM conversations
                WHERE user_id = :user_id
                  AND role    = 'user'
                  AND to_tsvector('english', content)
                      @@ to_tsquery('english', :tsquery)
                ORDER BY rank DESC
                LIMIT :limit
            """),
            {"user_id": user_id, "tsquery": ts_query, "limit": limit}
        )
        rows = result.fetchall()

        if not rows:
            # ILIKE fallback for short queries / tickers
            pattern = f"%{query_tokens[0]}%"
            result2 = await db.execute(
                text("""
                    SELECT content FROM conversations
                    WHERE user_id = :user_id
                      AND role    = 'user'
                      AND content ILIKE :pattern
                    ORDER BY created_at DESC
                    LIMIT :limit
                """),
                {"user_id": user_id, "pattern": pattern, "limit": limit // 2}
            )
            return [row[0] for row in result2.fetchall()]

        return [row[0] for row in rows]

    except Exception:
        return []


# ──────────────────────────────────────────────
#  Hybrid Search
# ──────────────────────────────────────────────

async def search_memory_hybrid(
    db: AsyncSession,
    user_id: int,
    query: str,
    top_k: int = 8
) -> List[str]:
    """
    Hybrid retrieval: BM25 (always) + pgvector (if embeddings available).
    BM25-only mode works well for financial queries.
    """
    import asyncio

    semantic_results, bm25_results = await asyncio.gather(
        search_memory_semantic(db, user_id, query, limit=10),
        search_memory_bm25(db, user_id, query, limit=10)
    )

    seen   = set()
    merged = []

    max_len = max(len(semantic_results), len(bm25_results))
    for i in range(max_len):
        if i < len(semantic_results):
            content = semantic_results[i]
            key = content[:80]
            if key not in seen:
                seen.add(key)
                merged.append(content)
        if i < len(bm25_results):
            content = bm25_results[i]
            key = content[:80]
            if key not in seen:
                seen.add(key)
                merged.append(content)

    return merged[:top_k]


# ──────────────────────────────────────────────
#  Recent History
# ──────────────────────────────────────────────

async def get_recent_history(
    db: AsyncSession,
    user_id: int,
    limit: int = 20
) -> List[dict]:
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(desc(Conversation.created_at))
        .limit(limit)
    )
    messages = list(result.scalars().all())
    messages.reverse()
    return [{"role": m.role, "content": m.content} for m in messages]


async def get_conversation_summary(
    db: AsyncSession,
    user_id: int,
    last_n: int = 50
) -> str:
    history = await get_recent_history(db, user_id, limit=last_n)
    if not history:
        return "No previous conversations."
    recent = history[-10:]
    lines  = []
    for msg in recent:
        prefix  = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)
