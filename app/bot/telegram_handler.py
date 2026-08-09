"""
Telegram webhook handler — speed optimized
─────────────────────────────────────────────────────────────────
Speed fixes:
  ✅ Continuous typing indicator (user feels instant response)
  ✅ asyncio.create_task for typing — non-blocking
  ✅ Typing cancels automatically when reply sends
  ✅ Fast document download (streaming)
"""
import asyncio
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from telegram import Bot, Message, Update

from app.bot.message_router import route_text_message
from app.bot.voice_handler import transcribe_voice
from app.config import settings
from app.models.user_repo import get_or_create_user

logger = logging.getLogger("finbot.handler")
bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)

MAX_FILE_BYTES = 20 * 1024 * 1024


async def _keep_typing(chat_id: int, stop_event: asyncio.Event) -> None:
    """
    Send typing action every 4s until stop_event is set.
    Telegram typing indicator lasts 5s — refresh every 4s.
    This makes bot feel "alive" during long processing.
    """
    while not stop_event.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action="typing")
        except Exception:
            pass
        try:
            await asyncio.wait_for(
                asyncio.shield(stop_event.wait()),
                timeout=4.0
            )
        except asyncio.TimeoutError:
            pass


async def handle_update(update_data: dict, db: AsyncSession) -> None:
    try:
        update  = Update.de_json(update_data, bot)
        message: Optional[Message] = update.effective_message
        if not message:
            return

        sender = message.from_user
        if not sender:
            return

        user, _ = await get_or_create_user(
            db, sender.id,
            username=sender.username,
            first_name=sender.first_name,
            last_name=sender.last_name,
        )
        chat_id = message.chat_id

        if message.text:
            await _handle_text(message, db, user, chat_id)

        elif message.voice or message.audio:
            await _handle_voice(message, db, user, chat_id)

        elif message.document:
            await _handle_document(message, db, user.id, chat_id)

        elif message.photo:
            await _send_message(
                chat_id,
                "📸 I can see your image. For best results, send PDFs or financial documents."
            )

    except Exception as e:
        logger.exception(f"Error handling update: {e}")


async def _handle_text(message: Message, db: AsyncSession, user, chat_id: int) -> None:
    """Handle text message with continuous typing indicator."""
    stop_event   = asyncio.Event()
    typing_task  = asyncio.create_task(_keep_typing(chat_id, stop_event))

    try:
        reply = await route_text_message(db, user, message.text)
    except Exception as e:
        logger.exception(f"route_text_message error: {e}")
        reply = "⚠️ Something went wrong. Please try again."
    finally:
        stop_event.set()
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

    await _send_message(chat_id, reply)


async def _handle_voice(message: Message, db: AsyncSession, user, chat_id: int) -> None:
    stop_event  = asyncio.Event()
    typing_task = asyncio.create_task(_keep_typing(chat_id, stop_event))

    try:
        transcript = await transcribe_voice(message, bot)
    finally:
        stop_event.set()
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

    if not transcript or transcript.startswith("Voice transcription failed"):
        await _send_message(
            chat_id,
            "⚠️ Couldn't transcribe that. Please try again or type your message."
        )
        return

    await _send_message(
        chat_id,
        f"🎙️ *I heard:* _{transcript}_",
        parse_mode="Markdown"
    )

    stop_event2  = asyncio.Event()
    typing_task2 = asyncio.create_task(_keep_typing(chat_id, stop_event2))
    try:
        reply = await route_text_message(db, user, transcript)
    except Exception as e:
        logger.exception(f"Voice route error: {e}")
        reply = "⚠️ Something went wrong. Please try again."
    finally:
        stop_event2.set()
        typing_task2.cancel()
        try:
            await typing_task2
        except asyncio.CancelledError:
            pass

    await _send_message(chat_id, reply)


async def _handle_document(
    message: Message, db: AsyncSession, user_id: int, chat_id: int
) -> None:
    doc      = message.document
    filename = doc.file_name or "document.pdf"

    if doc.file_size and doc.file_size > MAX_FILE_BYTES:
        await _send_message(chat_id, "⚠️ File too large. Please send documents under 20 MB.")
        return

    await _send_message(
        chat_id,
        f"📄 Processing *{filename}*… this may take a moment.",
        parse_mode="Markdown"
    )

    stop_event  = asyncio.Event()
    typing_task = asyncio.create_task(_keep_typing(chat_id, stop_event))

    try:
        file = await bot.get_file(doc.file_id)
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            response   = await client.get(
                f"https://api.telegram.org/file/bot{settings.TELEGRAM_BOT_TOKEN}/{file.file_path}"
            )
            file_bytes = response.content

        logger.info(f"Downloaded {filename}: {len(file_bytes)} bytes")

        if len(file_bytes) < 10:
            stop_event.set()
            await _send_message(chat_id, "⚠️ File appears to be empty. Please try uploading again.")
            return

        ext       = filename.lower().rsplit(".", 1)[-1] if "." in filename else "pdf"
        file_type = "pdf" if ext == "pdf" else "docx" if ext in ("docx", "doc") else "pdf"

        from app.services.document_service import DocumentService
        doc_service = DocumentService(db)
        saved       = await doc_service.process_document(
            user_id, filename, file_bytes, file_type=file_type
        )

        content_len = len(saved.content or "")
        logger.info(f"Document saved: id={saved.id}, chars={content_len}")

        if content_len < 50:
            reply = (
                f"⚠️ *{saved.filename}* uploaded but text extraction failed.\n\n"
                "This may happen with scanned/image-based PDFs. "
                "Try a text-based PDF or paste the text directly."
            )
        else:
            reply = (
                f"✅ *{saved.filename}* uploaded successfully.\n\n"
                f"*Summary:*\n{saved.summary}\n\n"
                "💬 Ask me anything about this document."
            )

    except Exception as e:
        logger.exception(f"Document processing failed: {e}")
        reply = "⚠️ Something went wrong. Please try again with a valid PDF."
    finally:
        stop_event.set()
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

    await _send_message(chat_id, reply, parse_mode="Markdown")


async def _send_message(
    chat_id: int, text: str, parse_mode: str = None
) -> None:
    max_len = 4096
    chunks  = [text[i:i + max_len] for i in range(0, len(text), max_len)]
    for chunk in chunks:
        try:
            await bot.send_message(
                chat_id=chat_id, text=chunk, parse_mode=parse_mode
            )
        except Exception as e:
            logger.warning(f"send_message failed ({parse_mode}): {e}")
            try:
                # Retry without parse_mode (formatting error ho sakta hai)
                await bot.send_message(chat_id=chat_id, text=chunk)
            except Exception:
                pass
