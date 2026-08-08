"""
Vectorless RAG Engine — BM25 + PostgreSQL Full-Text Search
"""
import re
import math
from typing import Optional
from collections import Counter
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


FINANCIAL_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "this", "that", "these", "those", "it", "its",
    "their", "they", "them", "what", "which", "who", "how", "when", "where",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "his", "her",
    "about", "tell", "me", "give", "show", "can", "please", "want",
    "need", "get", "find", "look", "check", "see", "know", "think"
}

FINANCIAL_PRESERVE = {
    "revenue", "profit", "loss", "earnings", "ebitda", "eps", "pe", "roe",
    "roa", "debt", "equity", "cash", "dividend", "margin", "growth", "risk",
    "share", "stock", "price", "market", "cap", "quarter", "annual", "fiscal",
    "fy", "q1", "q2", "q3", "q4", "crore", "billion", "million", "percent",
    "yoy", "qoq", "net", "gross", "operating", "total", "assets", "liabilities",
    "balance", "sheet", "income", "statement", "flow", "capex", "buyback",
    "acquisition", "merger", "ipo", "bond", "yield", "interest", "rate",
    "inflation", "gdp", "sector", "industry", "competitor", "guidance",
    "forecast", "outlook", "segment", "subsidiary", "consolidate", "filing",
    "10k", "10q", "8k", "sec", "audit", "compliance", "regulation", "subscribers",
    "arpu", "stores", "cities", "utilisation", "crude", "refinery", "telecom"
}

# ── Bug Fix: PostgreSQL ts_query unsafe characters ─────────────────
_TS_UNSAFE = re.compile(r"[&|!():*?'\"\\/\-\+\<\>\~\^\{\}\[\]]")


def sanitize_ts_token(token: str) -> str:
    """
    Remove all characters that are invalid in PostgreSQL tsquery.
    Returns empty string if nothing valid remains.
    """
    cleaned = _TS_UNSAFE.sub("", token).strip()
    # Must have alphabetic chars and be >= 2 chars
    if len(cleaned) < 2 or not any(c.isalpha() for c in cleaned):
        return ""
    # Pure numbers are meaningless in FTS
    if cleaned.isdigit():
        return ""
    return cleaned


def build_safe_ts_query(tokens: list[str]) -> str:
    """
    Build a safe PostgreSQL OR-joined ts_query from a list of tokens.
    Sanitizes every token — never crashes with special chars like & ? ! etc.
    """
    safe = [sanitize_ts_token(t) for t in tokens]
    safe = [t for t in safe if t]  # Remove empties
    if not safe:
        return ""
    return " | ".join(safe[:8])


def tokenize_financial(text_input: str) -> list[str]:
    text_lower = text_input.lower()
    tickers = re.findall(r'\b[A-Z]{2,6}\b', text_input)
    tokens = re.findall(r'\b[\w]+\b', text_lower)

    filtered = []
    for token in tokens:
        if token in FINANCIAL_PRESERVE:
            filtered.append(token)
        elif token not in FINANCIAL_STOP_WORDS and len(token) > 2:
            filtered.append(token)

    for ticker in tickers:
        t_lower = ticker.lower()
        if t_lower not in filtered:
            filtered.append(t_lower)

    return filtered


def bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    doc_freq: dict[str, int],
    total_docs: int,
    avg_doc_len: float,
    k1: float = 1.5,
    b: float = 0.75
) -> float:
    doc_len = len(doc_tokens)
    doc_token_count = Counter(doc_tokens)
    score = 0.0

    for token in query_tokens:
        if token not in doc_token_count:
            continue

        tf = doc_token_count[token]
        df = doc_freq.get(token, 1)
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
        tf_norm = (tf * (k1 + 1)) / (
            tf + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1))
        )
        score += idf * tf_norm

    return score


def chunk_document(
    content: str,
    chunk_size: int = 400,
    overlap: int = 80
) -> list[dict]:
    if not content:
        return []

    section_splits = re.split(
        r'\n(?=(?:Revenue|Profit|EBITDA|Risk|Segment|Business|Financial|'
        r'Key|Note|Outlook|Balance|Income|Cash|Capital|Debt|Equity|'
        r'Subscriber|Return|Market|\d+\.))',
        content,
        flags=re.IGNORECASE
    )

    chunks = []
    chunk_id = 0

    for section in section_splits:
        words = section.split()
        if not words:
            continue

        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)

            if len(chunk_text.strip()) > 30:
                tokens = tokenize_financial(chunk_text)
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "tokens": tokens,
                    "start_word": i,
                    "bm25_score": 0.0
                })
                chunk_id += 1

            i += chunk_size - overlap

    return chunks


class VectorlessRAG:
    def __init__(self, db: AsyncSession):
        self.db = db

    def retrieve_candidates(
        self,
        query: str,
        document_content: str,
        top_k: int = 15,
        filename: str = ""
    ) -> dict:
        if not document_content or not query:
            return {"chunks": [], "context": "", "query_tokens": []}

        query_tokens = tokenize_financial(query)
        if not query_tokens:
            query_tokens = [w.lower() for w in query.split() if len(w) > 2]

        chunks = chunk_document(document_content)
        if not chunks:
            return {
                "chunks": [],
                "context": document_content[:3000],
                "query_tokens": query_tokens
            }

        doc_freq: dict[str, int] = {}
        for chunk in chunks:
            for token in set(chunk["tokens"]):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        total_docs = len(chunks)
        avg_doc_len = sum(len(c["tokens"]) for c in chunks) / max(total_docs, 1)

        for chunk in chunks:
            score = bm25_score(
                query_tokens=query_tokens,
                doc_tokens=chunk["tokens"],
                doc_freq=doc_freq,
                total_docs=total_docs,
                avg_doc_len=avg_doc_len
            )

            has_numbers = bool(re.search(r'\d+[,.]?\d*', chunk["text"]))
            is_data_query = any(t in query_tokens for t in [
                "revenue", "profit", "ebitda", "margin", "growth", "debt",
                "earnings", "eps", "dividend", "crore", "billion", "percent",
                "subscribers", "arpu", "stores", "return", "equity"
            ])
            if has_numbers and is_data_query:
                score *= 1.35

            query_lower = query.lower()
            chunk_lower = chunk["text"].lower()
            query_words = query_lower.split()
            for j in range(len(query_words) - 1):
                phrase = query_words[j] + " " + query_words[j + 1]
                if phrase in chunk_lower:
                    score *= 1.2
                    break

            chunk["bm25_score"] = round(score, 4)

        sorted_chunks = sorted(chunks, key=lambda c: c["bm25_score"], reverse=True)
        candidates = [c for c in sorted_chunks[:top_k] if c["bm25_score"] > 0]

        if not candidates:
            candidates = chunks[:min(5, len(chunks))]
            for c in candidates:
                c["bm25_score"] = 0.01

        return {
            "chunks": candidates,
            "query_tokens": query_tokens,
            "total_chunks": total_docs,
            "candidate_count": len(candidates)
        }

    def build_context_from_chunks(self, chunks: list[dict]) -> str:
        if not chunks:
            return ""

        ordered = sorted(chunks, key=lambda c: c.get("id", 0))
        parts = []
        for i, chunk in enumerate(ordered):
            score_info = ""
            if chunk.get("reranked"):
                score_info = f" [relevance: {chunk.get('rerank_score', 0):.0f}/10]"
            parts.append(f"[Excerpt {i + 1}{score_info}]\n{chunk['text']}")

        return "\n\n".join(parts)

    def retrieve_from_document(
        self,
        query: str,
        document_content: str,
        top_k: int = 5,
        filename: str = ""
    ) -> dict:
        result = self.retrieve_candidates(query, document_content, top_k=top_k, filename=filename)
        chunks = result["chunks"][:top_k]
        context = self.build_context_from_chunks(chunks)
        return {
            "chunks": chunks,
            "context": context,
            "query_tokens": result["query_tokens"],
            "total_chunks": result.get("total_chunks", 0),
            "matched": len(chunks)
        }

    async def retrieve_from_conversations(
        self,
        user_id: int,
        query: str,
        top_k: int = 8
    ) -> list[str]:
        if not query:
            return []

        query_tokens = tokenize_financial(query)
        if not query_tokens:
            return []

        try:
            # ✅ Bug Fix: Use sanitized ts_query — no more crashes on & ? ! etc.
            ts_query = build_safe_ts_query(query_tokens)

            if ts_query:
                result = await self.db.execute(
                    text("""
                        SELECT content,
                               ts_rank(to_tsvector('english', content),
                                       to_tsquery('english', :tsquery)) AS rank
                        FROM conversations
                        WHERE user_id = :user_id
                          AND role = 'user'
                          AND to_tsvector('english', content) @@ to_tsquery('english', :tsquery)
                        ORDER BY rank DESC
                        LIMIT :limit
                    """),
                    {"user_id": user_id, "tsquery": ts_query, "limit": top_k * 2}
                )
                rows = result.fetchall()
            else:
                rows = []

            if not rows:
                # ILIKE fallback — always safe, no special chars
                safe_keyword = re.sub(r'[%_\\]', '', query_tokens[0]) if query_tokens else ""
                if safe_keyword:
                    result2 = await self.db.execute(
                        text("""
                            SELECT content FROM conversations
                            WHERE user_id = :user_id AND role = 'user'
                              AND content ILIKE :pattern
                            ORDER BY created_at DESC
                            LIMIT :limit
                        """),
                        {"user_id": user_id, "pattern": f"%{safe_keyword}%", "limit": top_k}
                    )
                    rows = result2.fetchall()

            if not rows:
                return []

            candidates = [{"text": row[0], "tokens": tokenize_financial(row[0])} for row in rows]
            doc_freq: dict[str, int] = {}
            for c in candidates:
                for t in set(c["tokens"]):
                    doc_freq[t] = doc_freq.get(t, 0) + 1

            avg_len = sum(len(c["tokens"]) for c in candidates) / max(len(candidates), 1)

            scored = []
            for c in candidates:
                score = bm25_score(
                    query_tokens=query_tokens,
                    doc_tokens=c["tokens"],
                    doc_freq=doc_freq,
                    total_docs=len(candidates),
                    avg_doc_len=avg_len
                )
                scored.append((score, c["text"]))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [t for _, t in scored[:top_k]]

        except Exception as e:
            # Log but never crash — return empty
            import logging
            logging.getLogger("finbot.rag").warning(f"FTS query failed: {e}")
            return []

    async def retrieve_from_memory_db(
        self,
        user_id: int,
        query: str,
        top_k: int = 4
    ) -> str:
        results = await self.retrieve_from_conversations(user_id, query, top_k=top_k * 2)
        if not results:
            return ""
        lines = []
        for i, content in enumerate(results[:top_k], 1):
            snippet = content[:250] + "..." if len(content) > 250 else content
            lines.append(f"[Memory {i}]: {snippet}")
        return "\n".join(lines)


async def setup_fts_indexes(db: AsyncSession) -> None:
    try:
        await db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_conversations_fts
            ON conversations
            USING GIN(to_tsvector('english', content))
        """))
        await db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_documents_fts
            ON documents
            USING GIN(to_tsvector('english',
                coalesce(content, '') || ' ' || coalesce(summary, '')))
        """))
        await db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_conversations_user_role
            ON conversations(user_id, role, created_at DESC)
        """))
        await db.commit()
    except Exception:
        await db.rollback()
