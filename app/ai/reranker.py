"""
LLM Reranker — openai/gpt-os-20b on Groq
─────────────────────────────────────────────────────────────────
Speed optimizations:
  ✅ openai/gpt-os-20b  (fastest Groq model, ~0.3s)
  ✅ max_tokens=60          (score array only — was 100)
  ✅ Skip reranking if chunks <= 3 (BM25 scores enough)
  ✅ Timeout=4s             (was 8s — fail fast, BM25 fallback)
  ✅ Memory reranking SKIPPED for simple/factual queries
"""
import json
import re

from groq import AsyncGroq
from app.config import settings

RERANKER_MODEL   = "openai/gpt-oss-20b"   # Fastest Groq model
RERANKER_TIMEOUT = 4.0                        # Fail fast → BM25 fallback

reranker_client = AsyncGroq(api_key=settings.GROQ_API_KEY)


async def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = 5,
    context_hint: str = ""
) -> list[dict]:
    """
    Rerank BM25 chunks with openai/gpt-oss-20b.
    Skip if chunks <= 3 (BM25 order is good enough).
    """
    if not chunks:
        return []

    # Fast path: <=3 chunks → skip reranker, use BM25 directly
    if len(chunks) <= 3:
        for c in chunks:
            c["rerank_score"] = c.get("bm25_score", 1.0)
            c["reranked"]     = False
        return sorted(chunks, key=lambda c: c.get("bm25_score", 0), reverse=True)[:top_k]

    # Build compact prompt — short previews save tokens → faster
    chunks_text = ""
    for i, chunk in enumerate(chunks):
        preview = chunk["text"][:200].replace("\n", " ").strip()  # 200 not 300
        chunks_text += f"[{i}]{preview}\n"

    prompt = (
        f'Score each chunk 0-10 for relevance to: "{query}"\n'
        f'{f"Context: {context_hint}" if context_hint else ""}\n'
        f"10=directly answers, 0=irrelevant.\n\n"
        f"{chunks_text}\n"
        f"Reply ONLY with JSON array: [score0, score1, ...]"
    )

    try:
        response = await reranker_client.chat.completions.create(
            model=RERANKER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.0,
            timeout=RERANKER_TIMEOUT
        )
        raw    = response.choices[0].message.content or "[]"
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
        # BM25 fallback — no delay
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
    """
    Rerank memories — skip if only 1-2 memories (not worth an API call).
    """
    if not memories:
        return []
    if len(memories) <= 2:
        return memories[:top_k]

    mem_text = "".join(
        f"[{i}]{mem[:150].replace(chr(10), ' ')}\n"
        for i, mem in enumerate(memories)
    )
    prompt = (
        f'Rate memory relevance 0-10 for query: "{query}"\n'
        f"{mem_text}\n"
        f"Reply ONLY with JSON array: [score0, score1, ...]"
    )

    try:
        response = await reranker_client.chat.completions.create(
            model=RERANKER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=40,
            temperature=0.0,
            timeout=RERANKER_TIMEOUT
        )
        raw    = response.choices[0].message.content or "[]"
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
