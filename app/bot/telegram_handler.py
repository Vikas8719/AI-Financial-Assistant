"""
Telegram webhook handler — Crash-proof version
─────────────────────────────────────────────────────────────────
Fixes applied:
  ✅ FIX 1: Bot instance lazy initialization (avoids startup crash)
  ✅ FIX 2: All exceptions caught — bot never crashes on any update
  ✅ FIX 3: Markdown parse error auto-retry without formatting
  ✅ FIX 4: Typing task properly cancelled before send (no dangling tasks)
  ✅ FIX 5: Empty/None text guard added
  ✅ FIX 6: Message chunking with safe split (no mid-word cuts)
  ✅ FIX 7: httpx timeout increased for large file downloads
  ✅ FIX 8: Document file_size None check fixed
"""
import asyncio
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from telegram import Bot, Message, Update
from telegram.error import TelegramError

from app.bot.message_router import route_text_message
from app.bot.voice_handler import transcribe_voice
from app.config import settings
from app.models.user_repo import get_or_create_user

logger = logging.getLogger("finbot.handler")

MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB
MAX_MSG_LEN    = 4096

# ── FIX 1: Lazy bot init — avoids crash if token is invalid at import time ──
_bot: Optional[Bot] = None

def get_bot() -> Bot:
    global _bot
    if _bot is None:
        _bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
    return _bot


# ──────────────────────────────────────────────────────────────
#  Typing indicator helper
# ──────────────────────────────────────────────────────────────

async def _keep_typing(chat_id: int, stop_event: asyncio.Event) -> None:
    """Send typing action every 4s until stop_event is set."""
    while not stop_event.is_set():
        try:
            await get_bot().send_chat_action(chat_id=chat_id, action="typing")
        except Exception:
            pass
        try:
            await asyncio.wait_for(
                asyncio.shield(stop_event.wait()),
                timeout=4.0
            )
        except asyncio.TimeoutError:
            pass


async def _run_with_typing(chat_id: int, coro):
    """
    Run coro while showing typing indicator.
    Returns (result, error_or_None).
    Always cancels typing cleanly — no dangling tasks.
    """
    stop_event  = asyncio.Event()
    typing_task = asyncio.create_task(_keep_typing(chat_id, stop_event))
    result, error = None, None
    try:
        result = await coro
    except Exception as e:
        error = e
    finally:
        stop_event.set()
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass
    return result, error


# ──────────────────────────────────────────────────────────────
#  Main update handler
# ──────────────────────────────────────────────────────────────

async def handle_update(update_data: dict, db: AsyncSession) -> None:
    """
    Entry point for all Telegram updates.
    FIX 2: Top-level try/except — bot NEVER crashes on any update.
    """
    try:
        update  = Update.de_json(update_data, get_bot())
        message: Optional[Message] = update.effective_message
        if not message:
            return

        sender = message.from_user
        if not sender:
            return

        user, _ = await get_or_create_user(
            db, sender.id,
            username   = sender.username,
            first_name = sender.first_name,
            last_name  = sender.last_name,
        )
        chat_id = message.chat_id

        # ── FIX 5: Guard empty text ──
        if message.text:
            text = message.text.strip()
            if text:
                await _handle_text(message, db, user, chat_id, text)

        elif message.voice or message.audio:
            await _handle_voice(message, db, user, chat_id)

        elif message.document:
            await _handle_document(message, db, user.id, chat_id)

        elif message.photo:
            await _send_message(
                chat_id,
                "📸 Image mili! PDF ya financial documents ke liye best results milte hain."
            )

    except Exception as e:
        # FIX 2: Log but NEVER re-raise — Telegram must always get 200 OK
        logger.exception(f"❌ handle_update crash prevented: {e}")


# ──────────────────────────────────────────────────────────────
#  Text handler
# ──────────────────────────────────────────────────────────────

async def _handle_text(
    message: Message, db: AsyncSession, user, chat_id: int, text: str
) -> None:
    result, error = await _run_with_typing(
        chat_id,
        route_text_message(db, user, text)
    )
    if error:
        logger.exception(f"route_text_message error: {error}")
        reply = "⚠️ Kuch galat ho gaya. Please dobara try karein."
    else:
        reply = result or "⚠️ Response generate nahi hua."

    await _send_message(chat_id, reply)


# ──────────────────────────────────────────────────────────────
#  Voice handler
# ──────────────────────────────────────────────────────────────

async def _handle_voice(
    message: Message, db: AsyncSession, user, chat_id: int
) -> None:
    transcript, t_error = await _run_with_typing(
        chat_id,
        transcribe_voice(message, get_bot())
    )

    if t_error or not transcript or transcript.startswith("Voice transcription failed"):
        await _send_message(
            chat_id,
            "⚠️ Audio samajh nahi aaya. Please dobara bolein ya type karein."
        )
        return

    await _send_message(
        chat_id,
        f"🎙️ *Suna:* _{transcript}_",
        parse_mode="Markdown"
    )

    result, r_error = await _run_with_typing(
        chat_id,
        route_text_message(db, user, transcript)
    )
    if r_error:
        logger.exception(f"Voice route error: {r_error}")
        reply = "⚠️ Kuch galat ho gaya. Please dobara try karein."
    else:
        reply = result or "⚠️ Response generate nahi hua."

    await _send_message(chat_id, reply)


# ──────────────────────────────────────────────────────────────
#  Document handler
# ──────────────────────────────────────────────────────────────

async def _handle_document(
    message: Message, db: AsyncSession, user_id: int, chat_id: int
) -> None:
    doc      = message.document
    filename = doc.file_name or "document.pdf"

    # ── FIX 8: file_size can be None ──
    file_size = doc.file_size or 0
    if file_size > MAX_FILE_BYTES:
        await _send_message(chat_id, "⚠️ File bahut badi hai. 20 MB se chhoti file bhejein.")
        return

    await _send_message(
        chat_id,
        f"📄 *{filename}* process ho raha hai… thoda wait karein.",
        parse_mode="Markdown"
    )

    async def _process():
        import httpx
        from app.services.document_service import DocumentService

        file_obj = await get_bot().get_file(doc.file_id)
        # ── FIX 7: Larger timeout for big files ──
        async with httpx.AsyncClient(timeout=60.0) as client:
            response   = await client.get(
                f"https://api.telegram.org/file/bot{settings.TELEGRAM_BOT_TOKEN}/{file_obj.file_path}"
            )
            file_bytes = response.content

        if len(file_bytes) < 10:
            return "⚠️ File empty lag rahi hai. Dobara upload karein."

        ext       = filename.lower().rsplit(".", 1)[-1] if "." in filename else "pdf"
        file_type = "pdf" if ext == "pdf" else "docx" if ext in ("docx", "doc") else "pdf"

        doc_service = DocumentService(db)
        saved       = await doc_service.process_document(
            user_id, filename, file_bytes, file_type=file_type
        )

        content_len = len(saved.content or "")
        if content_len < 50:
            return (
                f"⚠️ *{saved.filename}* upload to hua lekin text extract nahi ho saka.\n\n"
                "Scanned/image PDF hai? Text-based PDF try karein ya text paste karein."
            )
        return (
            f"✅ *{saved.filename}* successfully upload ho gaya.\n\n"
            f"*Summary:*\n{saved.summary}\n\n"
            "💬 Koi bhi sawaal poochh sakte hain is document ke baare mein."
        )

    result, error = await _run_with_typing(chat_id, _process())
    if error:
        logger.exception(f"Document processing failed: {error}")
        reply = "⚠️ Document process nahi hua. Valid PDF try karein."
    else:
        reply = result

    await _send_message(chat_id, reply, parse_mode="Markdown")


# ──────────────────────────────────────────────────────────────
#  Safe message sender
# ──────────────────────────────────────────────────────────────

async def _send_message(
    chat_id: int,
    text: str,
    parse_mode: Optional[str] = None
) -> None:
    """
    Send message with:
    - FIX 6: Safe chunking (split on newlines, not mid-word)
    - FIX 3: Auto-retry without parse_mode on Markdown errors
    """
    if not text or not text.strip():
        return

    chunks = _safe_split(text, MAX_MSG_LEN)
    bot    = get_bot()

    for chunk in chunks:
        if not chunk.strip():
            continue
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=chunk,
                parse_mode=parse_mode
            )
        except TelegramError as e:
            err_str = str(e).lower()
            # FIX 3: Markdown parse error → retry as plain text
            if parse_mode and (
                "can't parse" in err_str
                or "parse" in err_str
                or "entity" in err_str
                or "offset" in err_str
            ):
                logger.warning(f"Markdown parse error, retrying as plain text: {e}")
                try:
                    # Strip Markdown characters for safe plain text
                    plain = chunk.replace("*", "").replace("_", "").replace("`", "")
                    await bot.send_message(chat_id=chat_id, text=plain)
                except Exception as e2:
                    logger.error(f"Plain text fallback also failed: {e2}")
            else:
                logger.error(f"send_message failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected send_message error: {e}")


def _safe_split(text: str, max_len: int) -> list[str]:
    """
    FIX 6: Split long messages on newlines (not mid-word/mid-sentence).
    Falls back to hard split only if a single line exceeds max_len.
    """
    if len(text) <= max_len:
        return [text]

    chunks = []
    current = ""

    for line in text.split("\n"):
        # If adding this line would exceed limit, flush current chunk
        candidate = current + "\n" + line if current else line
        if len(candidate) > max_len:
            if current:
                chunks.append(current)
                current = line
            else:
                # Single line too long — hard split
                while len(line) > max_len:
                    chunks.append(line[:max_len])
                    line = line[max_len:]
                current = line
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks
