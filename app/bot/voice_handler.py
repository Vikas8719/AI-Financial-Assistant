"""
Voice message handler — downloads and transcribes Telegram voice notes.
"""
import logging

from telegram import Bot, Message

from app.services.voice_service import VoiceService

logger = logging.getLogger("finbot.voice")
voice_service = VoiceService()


async def transcribe_voice(message: Message, bot: Bot) -> str:
    """
    Download a voice/audio message from Telegram and return its transcript.
    Returns empty string on failure.
    """
    try:
        voice = message.voice or message.audio
        if not voice:
            return ""

        file = await bot.get_file(voice.file_id)
        file_bytes = await voice_service.download_telegram_file(
            file.file_path, bot.token
        )

        if not file_bytes:
            logger.warning("Downloaded 0 bytes for voice message")
            return ""

        transcript = await voice_service.transcribe(file_bytes, file.file_path)
        logger.info(f"Voice transcribed ({len(file_bytes)} bytes): {transcript[:80]}...")
        return transcript

    except Exception as e:
        logger.exception(f"Voice transcription error: {e}")
        return ""
