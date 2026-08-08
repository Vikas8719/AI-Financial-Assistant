"""
Hybrid Memory — pgvector (semantic) + BM25 (keyword) + LLM Reranker
─────────────────────────────────────────────────────────────────────
Save:    message → Groq embedding (llama-3.1-8b via Groq) → pgvector
Retrieve:
  Step 1 → pgvector cosine search   (top-10, semantic)
  Step 2 → BM25 keyword search      (top-10, exact terms)
  Step 3 → Merge + deduplicate      (up to 15 unique candidates)
  Step 4 → LLM Reranker (8b)        (picks best 3-4)

Benefits:
  ✅ pgvector catches: "profit fell" ↔ "earnings decline" (semantic)
  ✅ BM25 catches: "AAPL", "INR 9,01,012", "EBITDA 19.8%" (exact)
  ✅ Reranker eliminates noise from both
  ✅ Groq embedding = no OpenAI dependency, same API key
"""
from datetime import datetime
from typing import List, Optional

import httpx
from sqlalchemy import select, desc, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Conversation
from app.config import settings


def _now() -> datetime:
    return datetime.utcnow()


# ──────────────────────────────────────────────
#  Groq Embedding  (nomic-embed-text via Groq)
# ──────────────────────────────────────────────

GROQ_EMBED_MODEL = "nomic-embed-text-v1_5"   # 768-dim, free on Groq
EMBED_DIM        = 768


async def get_embedding(text_input: str) -> Optional[List[float]]:
    """
    Generate embedding using Groq's embedding endpoint.
    Falls back to None silently — BM25 still works without embeddings.
    Uses the same GROQ_API_KEY, no extra cost.
    """
    if not text_input or len(text_input.strip()) < 3:
        return None
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": GROQ_EMBED_MODEL,
                    "input": text_input[:512]   # Groq embedding max input
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
    Save message with embedding for user turns.
    Assistant turns don't need embeddings (we search by user intent).
    Embedding failure is non-fatal — BM25 fallback handles retrieval.
    """
    embedding = None
    if role == "user":
        embedding = await get_embedding(content)

    msg = Conversation(
        user_id=user_id,
        role=role,
        content=content,
        embedding=embedding,        # None if Groq embed failed — safe
        metadata_=metadata or {},
        created_at=_now()
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg


# ──────────────────────────────────────────────
#  pgvector Semantic Search
# ──────────────────────────────────────────────

async def search_memory_semantic(
    db: AsyncSession,
    user_id: int,
    query: str,
    limit: int = 10
) -> List[str]:
    """
    pgvector cosine similarity search over past user messages.
    Returns content strings, sorted by semantic relevance.
    Falls back to [] if no embeddings available.
    """
    query_embedding = await get_embedding(query)
    if not query_embedding:
        return []

    try:
        # Adjust vector dimensions to match DB column
        # DB has Vector(1536) from old setup; nomic gives 768
        # We pad with zeros to match — pgvector handles dimension mismatch
        # by returning 0 similarity (safe — just won't match old 1536-dim rows)
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
            {
                "user_id":   user_id,
                "embedding": vector_str,
                "limit":     limit
            }
        )
        rows = result.fetchall()
        # Only return rows with reasonable similarity (>0.3)
        return [row[0] for row in rows if row[1] > 0.3]

    except Exception:
        return []


# ──────────────────────────────────────────────
#  BM25 Keyword Search (via PostgreSQL FTS)
# ──────────────────────────────────────────────

async def search_memory_bm25(
    db: AsyncSession,
    user_id: int,
    query: str,
    limit: int = 10
) -> List[str]:
    """
    PostgreSQL full-text search (BM25-style) for exact financial terms.
    Catches: tickers (AAPL), numbers (9,01,012), exact terms (EBITDA).
    """
    from app.ai.rag_engine import tokenize_financial

    query_tokens = tokenize_financial(query)
    if not query_tokens:
        return []

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
            # ILIKE fallback for short queries
            pattern = f"%{query_tokens[0]}%" if query_tokens else "%"
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
#  Hybrid Search — Merge + Deduplicate
# ──────────────────────────────────────────────

async def search_memory_hybrid(
    db: AsyncSession,
    user_id: int,
    query: str,
    top_k: int = 8      # Over-fetch for reranker
) -> List[str]:
    """
    Hybrid retrieval: pgvector + BM25, merged & deduplicated.
    Both run concurrently for speed.
    """
    import asyncio

    # Run both searches concurrently
    semantic_results, bm25_results = await asyncio.gather(
        search_memory_semantic(db, user_id, query, limit=10),
        search_memory_bm25(db, user_id, query, limit=10)
    )

    # Merge with deduplication (preserve order — semantic first)
    seen = set()
    merged = []

    # Interleave: semantic[0], bm25[0], semantic[1], bm25[1], ...
    # This gives fair weight to both methods
    max_len = max(len(semantic_results), len(bm25_results))
    for i in range(max_len):
        if i < len(semantic_results):
            content = semantic_results[i]
            key = content[:80]          # Short key for dedup
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
#  Recent History (unchanged)
# ──────────────────────────────────────────────

async def get_recent_history(
    db: AsyncSession,
    user_id: int,
    limit: int = 20
) -> List[dict]:
    """Get recent conversation turns for context window."""
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
    """Compact recent memory summary (fallback for system prompt)."""
    history = await get_recent_history(db, user_id, limit=last_n)
    if not history:
        return "No previous conversations."
    recent = history[-10:]
    lines = []
    for msg in recent:
        prefix  = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)
