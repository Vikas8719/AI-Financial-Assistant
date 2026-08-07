"""Voice message processing using Groq Whisper API."""
import os
import tempfile

import httpx

from app.config import settings


class VoiceService:

    async def transcribe(self, file_bytes: bytes, filename: str = "audio.ogg") -> str:
        """Transcribe audio using Groq Whisper API."""
        try:
            suffix = os.path.splitext(filename)[1] or ".ogg"

            # Write to temp file synchronously (acceptable for small audio files)
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            try:
                from groq import AsyncGroq
                client = AsyncGroq(api_key=settings.GROQ_API_KEY)

                with open(tmp_path, "rb") as audio_file:
                    transcription = await client.audio.transcriptions.create(
                        file=(filename, audio_file, "audio/ogg"),
                        model="whisper-large-v3",
                        language="en",
                        response_format="text"
                    )

                return transcription if isinstance(transcription, str) else transcription.text

            finally:
                os.unlink(tmp_path)

        except Exception as e:
            return f"Voice transcription failed: {str(e)}"

    async def download_telegram_file(self, file_path: str, bot_token: str) -> bytes:
        """Download a file from Telegram servers."""
        url = f"https://api.telegram.org/file/bot{bot_token}/{file_path}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            return response.content
