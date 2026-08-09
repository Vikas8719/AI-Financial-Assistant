"""
LLM Reranker — with model fallback
FIX: openai/gpt-oss-20b rate limited → fallback to llama-3.1-8b-instant
     8b model is near-unlimited and fast enough for reranking (just scores)
"""
import json
import re
import logging

from groq import AsyncGroq, RateLimitError
from app.config import settings

logger = logging.getLogger("finbot.reranker")

RERANKER_PRIMARY = "openai/gpt-oss-20b"
RERANKER_FALLBACK = "llama-3.1-8b-instant"   # Fast, unlimited, good for scoring
RERANKER_TIMEOUT  = 4.0

reranker_client = AsyncGroq(api_key=settings.GROQ_API_KEY)


async def _reranker_call(prompt: str, max_tokens: int) -> str:
    """Try primary reranker → fallback on rate limit."""
    for model in [RERANKER_PRIMARY, RERANKER_FALLBACK]:
        try:
            response = await reranker_client.chat.completions.create(
                model       = model,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = max_tokens,
                temperature = 0.0,
                timeout     = RERANKER_TIMEOUT
            )
            return response.choices[0].message.content or "[]"
        except RateLimitError:
            if model == RERANKER_PRIMARY:
                logger.warning(f"Reranker {model} rate limited, trying {RERANKER_FALLBACK}")
                continue
            raise
        except Exception:
            raise
    return "[]"


async def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = 5,
    context_hint: str = ""
) -> list[dict]:
    """Rerank BM25 chunks. Skip if <=3 chunks — BM25 order is fine."""
    if not chunks:
        return []

    if len(chunks) <= 3:
        for c in chunks:
            c["rerank_score"] = c.get("bm25_score", 1.0)
            c["reranked"]     = False
        return sorted(chunks, key=lambda c: c.get("bm25_score", 0), reverse=True)[:top_k]

    chunks_text = ""
    for i, chunk in enumerate(chunks):
        preview = chunk["text"][:200].replace("\n", " ").strip()
        chunks_text += f"[{i}]{preview}\n"

    prompt = (
        f'Score 0-10 relevance to: "{query}"\n'
        f'{f"Context: {context_hint}" if context_hint else ""}\n'
        f"{chunks_text}\n"
        f"Reply ONLY: [score0, score1, ...]"
    )

    try:
        raw    = await _reranker_call(prompt, max_tokens=60)
        scores = _parse_score_array(raw, expected_len=len(chunks))

        for i, chunk in enumerate(chunks):
            bm25      = chunk.get("bm25_score", 0.0)
            rerank    = scores[i] if i < len(scores) else 0.0
            bm25_norm = min(bm25 / max(bm25 + 0.001, 1.0) * 10, 10)
            chunk["rerank_score"] = rerank
            chunk["hybrid_score"] = 0.3 * bm25_norm + 0.7 * rerank
            chunk["reranked"]     = True

        return sorted(chunks, key=lambda c: c.get("hybrid_score", 0), reverse=True)[:top_k]

    except Exception:
        # BM25 fallback — no API call needed
        for chunk in chunks:
            chunk["rerank_score"] = chunk.get("bm25_score", 0.0)
            chunk["hybrid_score"] = chunk.get("bm25_score", 0.0)
            chunk["reranked"]     = False
        return sorted(chunks, key=lambda c: c.get("bm25_score", 0), reverse=True)[:top_k]


async def rerank_memory(
    query: str,
    memories: list[str],
    top_k: int = 3
) -> list[str]:
    """Rerank memories. Skip if <=2."""
    if not memories:
        return []
    if len(memories) <= 2:
        return memories[:top_k]

    mem_text = "".join(
        f"[{i}]{mem[:150].replace(chr(10), ' ')}\n"
        for i, mem in enumerate(memories)
    )
    prompt = (
        f'Rate relevance 0-10 for: "{query}"\n'
        f"{mem_text}\n"
        f"Reply ONLY: [score0, score1, ...]"
    )

    try:
        raw    = await _reranker_call(prompt, max_tokens=40)
        scores = _parse_score_array(raw, expected_len=len(memories))
        scored = sorted(zip(scores, memories), key=lambda x: x[0], reverse=True)
        return [m for s, m in scored if s >= 4][:top_k]
    except Exception:
        return memories[:top_k]


def _parse_score_array(raw: str, expected_len: int) -> list[float]:
    raw = raw.strip()
    try:
        match = re.search(r'\[[\d\s.,]+\]', raw)
        if match:
            arr    = json.loads(match.group())
            scores = [float(x) for x in arr]
            while len(scores) < expected_len:
                scores.append(0.0)
            return scores[:expected_len]
    except Exception:
        pass
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', raw)
    scores  = [float(n) for n in numbers if float(n) <= 10]
    while len(scores) < expected_len:
        scores.append(0.0)
    return scores[:expected_len]
