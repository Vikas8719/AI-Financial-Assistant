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
        summary = await self._generate_summary(content, filename)

        doc = Document(
            user_id=user_id,
            filename=filename,
            file_type=file_type,
            content=content[:50000],  # Limit stored content
            summary=summary,
            metadata_={"size": len(file_bytes), "char_count": len(content)}
        )
        self.db.add(doc)
        await self.db.commit()
        await self.db.refresh(doc)
        return doc

    async def _extract_text(self, file_bytes: bytes, file_type: str, filename: str) -> str:
        """Extract text from PDF or Word document."""
        try:
            if file_type == "pdf" or filename.lower().endswith(".pdf"):
                return await self._extract_pdf(file_bytes)
            elif filename.lower().endswith((".docx", ".doc")):
                return await self._extract_docx(file_bytes)
            else:
                # Try as plain text
                return file_bytes.decode("utf-8", errors="ignore")
        except Exception as e:
            return f"Could not extract text: {str(e)}"

    async def _extract_pdf(self, file_bytes: bytes) -> str:
        """Extract text from PDF using pdfplumber."""
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                text_parts = []
                for page in pdf.pages[:50]:  # Max 50 pages
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                return "\n\n".join(text_parts)
        except Exception:
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                text = ""
                for page in doc:
                    text += page.get_text()
                return text
            except Exception as e:
                return f"PDF extraction failed: {str(e)}"

    async def _extract_docx(self, file_bytes: bytes) -> str:
        """Extract text from Word document."""
        try:
            from docx import Document as DocxDocument
            doc = DocxDocument(io.BytesIO(file_bytes))
            return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        except Exception as e:
            return f"DOCX extraction failed: {str(e)}"

    async def _generate_summary(self, content: str, filename: str) -> str:
        """Generate a structured summary of the document."""
        if not content or len(content) < 100:
            return "Document appears to be empty or could not be parsed."

        # Use first 8000 chars for summary
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
            return response.choices[0].message.content
        except Exception as e:
            return f"Summary generation failed: {str(e)}"

    async def answer_question(
        self,
        user_id: int,
        question: str,
        document_id: Optional[int] = None
    ) -> dict:
        """Answer a question about a document."""
        # Get document(s)
        if document_id:
            result = await self.db.execute(
                select(Document).where(Document.id == document_id, Document.user_id == user_id)
            )
            doc = result.scalar_one_or_none()
            docs = [doc] if doc else []
        else:
            # Get most recent document
            result = await self.db.execute(
                select(Document)
                .where(Document.user_id == user_id)
                .order_by(Document.created_at.desc())
                .limit(1)
            )
            docs = result.scalars().all()

        if not docs:
            return {"answer": "No document found. Please upload a document first.", "found": False}

        doc = docs[0]
        content_excerpt = (doc.content or "")[:12000]

        try:
            response = await client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial document analyst. Answer questions accurately based only on the provided document content. If information is not in the document, say so clearly."
                    },
                    {
                        "role": "user",
                        "content": f"Document: {doc.filename}\n\nContent:\n{content_excerpt}\n\nQuestion: {question}"
                    }
                ],
                max_tokens=800,
                temperature=0.1
            )
            answer = response.choices[0].message.content
            return {
                "answer": answer,
                "document": doc.filename,
                "found": True
            }
        except Exception as e:
            return {"error": str(e), "found": True}

    async def get_user_documents(self, user_id: int) -> list:
        """Get list of user's uploaded documents."""
        result = await self.db.execute(
            select(Document)
            .where(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
            .limit(10)
        )
        docs = result.scalars().all()
        return [{"id": d.id, "filename": d.filename, "type": d.file_type, "uploaded": d.created_at.isoformat()} for d in docs]
