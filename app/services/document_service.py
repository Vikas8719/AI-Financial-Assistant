"""Document Intelligence — PDF parsing, summarization, Q&A."""
import io
import os
import tempfile
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from groq import AsyncGroq

from app.database import Document
from app.config import settings

client = AsyncGroq(api_key=settings.GROQ_API_KEY)


class DocumentService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def process_document(
        self,
        user_id: int,
        filename: str,
        file_bytes: bytes,
        file_type: str = "pdf"
    ) -> Document:
        """Extract text from document and save to DB."""
        content = await self._extract_text(file_bytes, file_type, filename)

        if not content or len(content.strip()) < 50:
            # Last resort: try raw decode
            try:
                content = file_bytes.decode("utf-8", errors="ignore")
            except Exception:
                content = ""

        summary = await self._generate_summary(content, filename)

        doc = Document(
            user_id=user_id,
            filename=filename,
            file_type=file_type,
            content=content[:50000],
            summary=summary,
            metadata_={"size": len(file_bytes), "char_count": len(content)}
        )
        self.db.add(doc)
        await self.db.commit()
        await self.db.refresh(doc)
        return doc

    async def _extract_text(self, file_bytes: bytes, file_type: str, filename: str) -> str:
        """Try multiple PDF extraction methods."""
        text = ""

        # Method 1: PyMuPDF (most reliable)
        try:
            import fitz
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            parts = []
            for page in doc:
                t = page.get_text()
                if t:
                    parts.append(t)
            doc.close()
            text = "\n\n".join(parts)
            if len(text.strip()) > 100:
                return text
        except Exception as e:
            pass

        # Method 2: pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                parts = []
                for page in pdf.pages[:50]:
                    t = page.extract_text()
                    if t:
                        parts.append(t)
                text = "\n\n".join(parts)
                if len(text.strip()) > 100:
                    return text
        except Exception:
            pass

        # Method 3: Raw text extraction (for text-based PDFs)
        try:
            raw = file_bytes.decode("latin-1", errors="ignore")
            # Extract text between stream markers
            import re
            streams = re.findall(r'BT(.*?)ET', raw, re.DOTALL)
            extracted = []
            for stream in streams:
                words = re.findall(r'\((.*?)\)', stream)
                if words:
                    extracted.extend(words)
            if extracted:
                text = " ".join(extracted)
                if len(text.strip()) > 50:
                    return text
        except Exception:
            pass

        # Method 4: docx
        if filename.lower().endswith((".docx", ".doc")):
            try:
                from docx import Document as DocxDocument
                doc = DocxDocument(io.BytesIO(file_bytes))
                text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                if len(text.strip()) > 50:
                    return text
            except Exception:
                pass

        return text or ""

    async def _generate_summary(self, content: str, filename: str) -> str:
        """Generate a structured summary of the document."""
        if not content or len(content.strip()) < 50:
            return "Document could not be parsed. Please try uploading again or use a text-based PDF."

        excerpt = content[:8000]
        try:
            response = await client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{
                    "role": "user",
                    "content": f"""Analyze this financial document '{filename}' and provide:
1. Document type and purpose (1 sentence)
2. Key financial metrics or figures mentioned
3. Top 3-5 insights or highlights
4. Any risks or concerns mentioned
5. Time period covered

Document content:
{excerpt}

Be concise and focused on what a finance professional needs to know."""
                }],
                max_tokens=600,
                temperature=0.2
            )
            return response.choices[0].message.content or "Summary not available."
        except Exception as e:
            return f"Summary generation failed: {str(e)}"

    async def answer_question(
        self,
        user_id: int,
        question: str,
        document_id: Optional[int] = None
    ) -> dict:
        """Answer a question about a document."""
        if document_id:
            result = await self.db.execute(
                select(Document).where(
                    Document.id == document_id,
                    Document.user_id == user_id
                )
            )
            doc = result.scalar_one_or_none()
        else:
            result = await self.db.execute(
                select(Document)
                .where(Document.user_id == user_id)
                .order_by(Document.created_at.desc())
                .limit(1)
            )
            doc = result.scalars().first()

        if not doc:
            return {"answer": "No document found. Please upload a document first.", "found": False}

        content_excerpt = (doc.content or "")[:12000]

        try:
            response = await client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a financial document analyst. "
                            "Answer questions accurately based ONLY on the provided document content. "
                            "If information is in the document, answer directly with specific numbers and facts. "
                            "If not found, say so clearly."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Document: {doc.filename}\n\nContent:\n{content_excerpt}\n\nQuestion: {question}"
                    }
                ],
                max_tokens=800,
                temperature=0.1
            )
            return {
                "answer": response.choices[0].message.content,
                "document": doc.filename,
                "found": True
            }
        except Exception as e:
            return {"error": str(e), "found": True}

    async def get_user_documents(self, user_id: int) -> list:
        result = await self.db.execute(
            select(Document)
            .where(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
            .limit(10)
        )
        docs = result.scalars().all()
        return [
            {
                "id": d.id,
                "filename": d.filename,
                "type": d.file_type,
                "uploaded": d.created_at.isoformat()
            }
            for d in docs
        ]
