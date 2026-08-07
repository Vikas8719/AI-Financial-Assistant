"""pgvector-powered conversation memory with semantic search."""
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Conversation
from app.config import settings
import httpx


async def get_embedding(text_input: str) -> Optional[List[float]]:
    """Get text embedding — returns None if unavailable."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
                json={"input": text_input, "model": "text-embedding-3-small"},
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()["data"][0]["embedding"]
    except Exception:
        pass
    return None


async def save_message(
    db: AsyncSession,
    user_id: int,
    role: str,
    content: str,
    metadata: Optional[dict] = None
) -> Conversation:
    """Save a conversation message with optional embedding."""
    embedding = await get_embedding(content) if role == "user" else None

    msg = Conversation(
        user_id=user_id,
        role=role,
        content=content,
        embedding=embedding,
        metadata_=metadata or {},
        created_at=datetime.now(timezone.utc)
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg


async def get_recent_history(
    db: AsyncSession,
    user_id: int,
    limit: int = 20
) -> List[dict]:
    """Get recent conversation history for context window."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(desc(Conversation.created_at))
        .limit(limit)
    )
    messages = list(result.scalars().all())
    messages.reverse()
    return [{"role": m.role, "content": m.content} for m in messages]


async def search_memory(
    db: AsyncSession,
    user_id: int,
    query: str,
    limit: int = 5
) -> List[str]:
    """Semantic search over past conversations using pgvector."""
    from sqlalchemy import text
    query_embedding = await get_embedding(query)
    if not query_embedding:
        return []

    try:
        vector_str = f"[{','.join(map(str, query_embedding))}]"
        result = await db.execute(
            text("""
                SELECT content FROM conversations
                WHERE user_id = :user_id AND embedding IS NOT NULL
                ORDER BY embedding <=> :embedding::vector
                LIMIT :limit
            """),
            {"user_id": user_id, "embedding": vector_str, "limit": limit}
        )
        return [row[0] for row in result.fetchall()]
    except Exception:
        return []


async def get_conversation_summary(
    db: AsyncSession,
    user_id: int,
    last_n: int = 50
) -> str:
    """Get a compact summary string of recent memory for system prompt."""
    history = await get_recent_history(db, user_id, limit=last_n)
    if not history:
        return "No previous conversations."

    recent = history[-10:]
    lines = []
    for msg in recent:
        prefix = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        lines.append(f"{prefix}: {content}")

    return "\n".join(lines)
