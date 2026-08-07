"""Document Intelligence — PDF parsing, summarization, Q&A."""
import io
import re
from typing import Optional

from groq import AsyncGroq
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import Document

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
        content = await self._extract_text(file_bytes, filename)
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

    async def _extract_text(self, file_bytes: bytes, filename: str) -> str:
        """Try every possible method to extract text."""
        text = ""

        # Method 1: PyMuPDF
        try:
            import fitz
            with fitz.open(stream=file_bytes, filetype="pdf") as pdf_doc:
                pages = [page.get_text() for page in pdf_doc]
                text = "\n\n".join(p for p in pages if p.strip())
            if len(text.strip()) > 100:
                return text
        except Exception as e:
            pass

        # Method 2: pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages[:50]]
                text = "\n\n".join(p for p in pages if p.strip())
            if len(text.strip()) > 100:
                return text
        except Exception:
            pass

        # Method 3: docx
        if filename.lower().endswith((".docx", ".doc")):
            try:
                from docx import Document as DocxDocument
                doc = DocxDocument(io.BytesIO(file_bytes))
                text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                if len(text.strip()) > 50:
                    return text
            except Exception:
                pass

        # Method 4: Raw PDF text extraction
        try:
            raw = file_bytes.decode("latin-1", errors="ignore")
            # Extract readable text between parentheses (PDF text encoding)
            chunks = re.findall(r'\(((?:[^()\\]|\\[\s\S])*)\)', raw)
            readable = []
            for chunk in chunks:
                # Unescape PDF escape sequences
                chunk = chunk.replace("\\n", "\n").replace("\\r", "\r")
                chunk = chunk.replace("\\t", "\t").replace("\\\\", "\\")
                # Only keep chunks with readable ASCII
                if len(chunk) > 2 and any(c.isalpha() for c in chunk):
                    readable.append(chunk)
            text = " ".join(readable)
            if len(text.strip()) > 100:
                return text
        except Exception:
            pass

        # Method 5: UTF-8 decode as last resort
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
            # Remove binary garbage, keep readable parts
            clean = re.sub(r'[^\x20-\x7E\n\r\t]', ' ', text)
            clean = re.sub(r' {3,}', ' ', clean)
            # Only return if we have meaningful content
            words = [w for w in clean.split() if len(w) > 2 and w.isalpha()]
            if len(words) > 20:
                return clean
        except Exception:
            pass

        return ""

    async def _generate_summary(self, content: str, filename: str) -> str:
        if not content or len(content.strip()) < 50:
            return "⚠️ Could not extract text from this document. Try a text-based PDF."

        excerpt = content[:8000]
        try:
            response = await client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{
                    "role": "user",
                    "content": f"""Analyze this financial document '{filename}':

{excerpt}

Provide:
1. Document type and purpose (1 sentence)
2. Key financial metrics with exact numbers
3. Top 3-5 insights
4. Risks mentioned
5. Time period covered

Be concise and use exact figures from the document."""
                }],
                max_tokens=700,
                temperature=0.1
            )
            return response.choices[0].message.content or "Summary unavailable."
        except Exception as e:
            return f"Summary generation failed: {str(e)}"

    async def answer_question(
        self,
        user_id: int,
        question: str,
        document_id: Optional[int] = None
    ) -> dict:
        if document_id:
            result = await self.db.execute(
                select(Document).where(Document.id == document_id, Document.user_id == user_id)
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

        if len(content_excerpt.strip()) < 50:
            return {
                "answer": "⚠️ The document content could not be extracted. Please re-upload the PDF.",
                "found": True
            }

        try:
            response = await client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a financial document analyst. "
                            "Answer questions using ONLY the provided document content. "
                            "Give specific numbers and facts. "
                            "If the answer is in the document, state it directly."
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
        return [{"id": d.id, "filename": d.filename, "type": d.file_type} for d in docs]
