"""
Telegram webhook handler.
Receives every Update, dispatches to the right handler.
"""
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from telegram import Bot, Message, Update

from app.bot.message_router import route_text_message
from app.bot.voice_handler import transcribe_voice
from app.config import settings
from app.models.user_repo import get_or_create_user
from app.services.document_service import DocumentService
from app.services.voice_service import VoiceService

logger = logging.getLogger("finbot.handler")
bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
voice_service = VoiceService()

# Max file size for documents (20 MB)
MAX_FILE_BYTES = 20 * 1024 * 1024


async def handle_update(update_data: dict, db: AsyncSession) -> None:
    """Entry point for every Telegram update."""
    try:
        update = Update.de_json(update_data, bot)
        message: Optional[Message] = update.effective_message
        if not message:
            return

        sender = message.from_user
        if not sender:
            return

        user, created = await get_or_create_user(
            db,
            sender.id,
            username=sender.username,
            first_name=sender.first_name,
            last_name=sender.last_name,
        )
        chat_id = message.chat_id

        # ── Text ─────────────────────────────────────────────
        if message.text:
            await _send_typing(chat_id)
            reply = await route_text_message(db, user, message.text)
            await _send_message(chat_id, reply)

        # ── Voice / Audio ─────────────────────────────────────
        elif message.voice or message.audio:
            await _handle_voice(message, db, user, chat_id)

        # ── Document / PDF ────────────────────────────────────
        elif message.document:
            await _handle_document(message, db, user.id, chat_id)

        # ── Photo ─────────────────────────────────────────────
        elif message.photo:
            await _send_message(
                chat_id,
                "📸 I can see your image. Right now I work best with PDFs and financial documents. "
                "Feel free to ask me anything in text."
            )

        else:
            await _send_message(
                chat_id,
                "Send me a text message, a voice note, or a PDF/document and I'll help you out."
            )

    except Exception as e:
        logger.exception(f"Error handling update: {e}")


async def _handle_voice(
    message: Message, db: AsyncSession, user, chat_id: int
) -> None:
    await _send_typing(chat_id)
    transcript = await transcribe_voice(message, bot)

    if not transcript or transcript.startswith("Voice transcription failed"):
        await _send_message(
            chat_id,
            "⚠️ Sorry, I couldn't transcribe that voice message. "
            "Please try again or type your message."
        )
        return

    # Echo transcript so user knows what was heard
    await _send_message(chat_id, f"🎙️ *I heard:* _{transcript}_", parse_mode="Markdown")

    # Route transcribed text through the AI agent
    await _send_typing(chat_id)
    reply = await route_text_message(db, user, transcript)
    await _send_message(chat_id, reply)


async def _handle_document(
    message: Message, db: AsyncSession, user_id: int, chat_id: int
) -> None:
    doc = message.document
    filename = doc.file_name or "document"

    # Reject oversized files early
    if doc.file_size and doc.file_size > MAX_FILE_BYTES:
        await _send_message(
            chat_id,
            f"⚠️ File too large ({doc.file_size // (1024*1024)} MB). "
            "Please send documents under 20 MB."
        )
        return

    await _send_message(chat_id, f"📄 Processing *{filename}*… this may take a moment.", parse_mode="Markdown")

    try:
        file = await bot.get_file(doc.file_id)
        file_bytes = await voice_service.download_telegram_file(
            file.file_path, settings.TELEGRAM_BOT_TOKEN
        )

        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        file_type = "pdf" if ext == "pdf" else "docx" if ext in ("docx", "doc") else "pdf"

        doc_service = DocumentService(db)
        saved = await doc_service.process_document(user_id, filename, file_bytes, file_type=file_type)

        reply = (
            f"✅ *{saved.filename}* uploaded successfully.\n\n"
            f"*Summary:*\n{saved.summary}\n\n"
            "💬 Ask me anything about this document."
        )
        await _send_message(chat_id, reply, parse_mode="Markdown")

    except Exception as e:
        logger.exception(f"Document processing failed: {e}")
        await _send_message(
            chat_id,
            "⚠️ Something went wrong while processing your document. "
            "Please make sure it's a valid PDF or Word file and try again."
        )


async def _send_message(chat_id: int, text: str, parse_mode: str = None) -> None:
    """Send message, splitting if it exceeds Telegram's 4096-char limit."""
    max_len = 4096
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)]
    for chunk in chunks:
        try:
            await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=parse_mode)
        except Exception as e:
            logger.warning(f"send_message failed (parse_mode={parse_mode}): {e}")
            # Retry without parse_mode to avoid formatting errors crashing the bot
            if parse_mode:
                try:
                    await bot.send_message(chat_id=chat_id, text=chunk)
                except Exception:
                    pass


async def _send_typing(chat_id: int) -> None:
    try:
        await bot.send_chat_action(chat_id=chat_id, action="typing")
    except Exception:
        pass
