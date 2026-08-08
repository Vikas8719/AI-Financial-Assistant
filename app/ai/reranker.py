"""
LLM Reranker — llama-3.1-8b-instant on Groq
─────────────────────────────────────────────
Pipeline:
  1. BM25 retrieves top-15 candidate chunks (fast, no API call)
  2. llama-3.1-8b-instant reranks them (tiny model, 1-2s, near-zero tokens)
  3. Top-5 go to main 70b LLM as pinned context

Why separate model for reranking?
  - 8b is 10x faster + cheaper than 70b for scoring tasks
  - Reranking is classification, not generation — 8b handles it perfectly
  - Saves 70b tokens for actual answer generation
  - Total latency: BM25(0ms) + 8b-rerank(1s) + 70b-answer(2s) = 3s vs old 5s
"""
import json
import re
from typing import Optional

from groq import AsyncGroq

from app.config import settings

# ── Model config ──────────────────────────────────────────────
RERANKER_MODEL = "llama-3.1-8b-instant"   # Fast, cheap reranker
MAIN_MODEL = "llama-3.3-70b-versatile"    # Main answer model

# Safety: if Groq rate-limits the reranker, fall back to BM25 scores
RERANKER_TIMEOUT = 8.0  # seconds

reranker_client = AsyncGroq(api_key=settings.GROQ_API_KEY)


# ──────────────────────────────────────────────────────────────
#  Reranker — pointwise scoring
# ──────────────────────────────────────────────────────────────

async def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = 5,
    context_hint: str = ""
) -> list[dict]:
    """
    Rerank BM25 candidate chunks using llama-3.1-8b-instant.

    Scoring strategy: pointwise — model assigns 0-10 relevance score
    to each chunk independently. Fast, parallelisable, deterministic.

    Args:
        query       : user's original question
        chunks      : list of {"id", "text", "bm25_score"} dicts
        top_k       : how many top chunks to return
        context_hint: extra hint (e.g. "user is asking about revenue")

    Returns:
        Sorted list of chunks with added "rerank_score" field
    """
    if not chunks:
        return []

    # If only 1-2 chunks, reranking adds no value — skip it
    if len(chunks) <= 2:
        for c in chunks:
            c["rerank_score"] = c.get("bm25_score", 1.0)
            c["reranked"] = False
        return chunks[:top_k]

    # Build batch scoring prompt — single call for all chunks
    # (one call is better than N parallel calls for rate limit safety)
    chunks_text = ""
    for i, chunk in enumerate(chunks):
        text_preview = chunk["text"][:300].replace("\n", " ").strip()
        chunks_text += f"\n[CHUNK {i}]\n{text_preview}\n"

    prompt = f"""You are a financial document relevance scorer.

QUERY: "{query}"
{f'CONTEXT: {context_hint}' if context_hint else ''}

Score each chunk's relevance to the query on a scale of 0-10:
- 10: Directly answers the query with exact data/numbers
- 7-9: Highly relevant, contains related financial information
- 4-6: Partially relevant, tangentially related
- 1-3: Mostly irrelevant but has minor overlap
- 0: Completely irrelevant

CHUNKS TO SCORE:
{chunks_text}

Respond ONLY with a JSON array of scores in order, example:
[8, 3, 9, 1, 6, 0, 7, 4]

JSON scores array:"""

    try:
        response = await reranker_client.chat.completions.create(
            model=RERANKER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,          # Just a score array — very short
            temperature=0.0,         # Deterministic scoring
            timeout=RERANKER_TIMEOUT
        )

        raw = response.choices[0].message.content or "[]"
        scores = _parse_score_array(raw, expected_len=len(chunks))

        # Attach rerank scores
        for i, chunk in enumerate(chunks):
            bm25 = chunk.get("bm25_score", 0.0)
            rerank = scores[i] if i < len(scores) else 0.0

            # Hybrid score: 30% BM25 + 70% LLM rerank
            # BM25 normalised to 0-10 scale for fair weighting
            bm25_norm = min(bm25 / max(bm25 + 0.001, 1.0) * 10, 10)
            hybrid = 0.3 * bm25_norm + 0.7 * rerank

            chunk["rerank_score"] = rerank
            chunk["hybrid_score"] = hybrid
            chunk["reranked"] = True

        # Sort by hybrid score
        ranked = sorted(chunks, key=lambda c: c.get("hybrid_score", 0), reverse=True)
        return ranked[:top_k]

    except Exception:
        # Fallback: return BM25 order unchanged
        for chunk in chunks:
            chunk["rerank_score"] = chunk.get("bm25_score", 0.0)
            chunk["hybrid_score"] = chunk.get("bm25_score", 0.0)
            chunk["reranked"] = False
        return sorted(chunks, key=lambda c: c.get("bm25_score", 0), reverse=True)[:top_k]


async def rerank_memory(
    query: str,
    memories: list[str],
    top_k: int = 3
) -> list[str]:
    """
    Rerank past conversation memories using llama-3.1-8b-instant.
    Keeps only genuinely relevant memories for context injection.

    Args:
        query   : current user query
        memories: list of past conversation snippets
        top_k   : how many to keep

    Returns:
        Filtered & sorted list of relevant memories
    """
    if not memories or len(memories) <= 1:
        return memories[:top_k]

    mem_text = ""
    for i, mem in enumerate(memories):
        preview = mem[:200].replace("\n", " ").strip()
        mem_text += f"\n[MEM {i}]\n{preview}\n"

    prompt = f"""You are filtering conversation memory for relevance.

CURRENT QUERY: "{query}"

Rate each memory's relevance (0-10):
- 10: Directly related, user mentioned this topic before
- 5-9: Somewhat related context
- 1-4: Loosely related
- 0: Completely unrelated

MEMORIES:
{mem_text}

Respond ONLY with JSON scores array, example: [7, 0, 4]

Scores:"""

    try:
        response = await reranker_client.chat.completions.create(
            model=RERANKER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.0,
            timeout=RERANKER_TIMEOUT
        )
        raw = response.choices[0].message.content or "[]"
        scores = _parse_score_array(raw, expected_len=len(memories))

        # Pair memories with scores and filter
        scored = list(zip(scores, memories))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Only keep memories with score >= 4 (relevant threshold)
        relevant = [mem for score, mem in scored if score >= 4]
        return relevant[:top_k]

    except Exception:
        return memories[:top_k]


# ──────────────────────────────────────────────────────────────
#  Helper: parse LLM score output
# ──────────────────────────────────────────────────────────────

def _parse_score_array(raw: str, expected_len: int) -> list[float]:
    """
    Robustly parse score array from LLM output.
    Handles various formats the model might output.
    """
    raw = raw.strip()

    # Try direct JSON array parse
    try:
        match = re.search(r'\[[\d\s.,]+\]', raw)
        if match:
            arr = json.loads(match.group())
            if isinstance(arr, list):
                # Pad or trim to expected length
                scores = [float(x) for x in arr]
                while len(scores) < expected_len:
                    scores.append(0.0)
                return scores[:expected_len]
    except Exception:
        pass

    # Fallback: extract all numbers
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', raw)
    scores = [float(n) for n in numbers if float(n) <= 10]

    # Pad if short
    while len(scores) < expected_len:
        scores.append(0.0)

    return scores[:expected_len]
