"""Google OAuth + Gmail + Calendar integration."""
import json
from typing import Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import User
from app.config import settings


class GoogleService:
    def __init__(self, user_id: int, db: AsyncSession):
        self.user_id = user_id
        self.db = db

    async def _get_credentials(self):
        """Get stored Google credentials for user."""
        result = await self.db.execute(
            select(User).where(User.id == self.user_id)
        )
        user = result.scalar_one_or_none()
        if not user or not user.google_tokens:
            return None
        return user.google_tokens

    async def search_emails(self, query: str, max_results: int = 5) -> dict:
        """Search Gmail for emails matching query."""
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            tokens = await self._get_credentials()
            if not tokens:
                return {"error": "Google account not connected. Use /connect to link your Gmail.", "connected": False}

            creds = Credentials(
                token=tokens.get("access_token"),
                refresh_token=tokens.get("refresh_token"),
                token_uri="https://oauth2.googleapis.com/token",
                client_id=settings.GOOGLE_CLIENT_ID,
                client_secret=settings.GOOGLE_CLIENT_SECRET
            )

            service = build("gmail", "v1", credentials=creds)
            result = service.users().messages().list(
                userId="me",
                q=query,
                maxResults=max_results
            ).execute()

            messages = result.get("messages", [])
            emails = []
            for msg in messages[:5]:
                msg_data = service.users().messages().get(
                    userId="me",
                    id=msg["id"],
                    format="metadata",
                    metadataHeaders=["From", "Subject", "Date"]
                ).execute()

                headers = {h["name"]: h["value"] for h in msg_data.get("payload", {}).get("headers", [])}
                snippet = msg_data.get("snippet", "")
                emails.append({
                    "from": headers.get("From", ""),
                    "subject": headers.get("Subject", ""),
                    "date": headers.get("Date", ""),
                    "snippet": snippet[:200]
                })

            return {"query": query, "emails": emails, "total": len(messages), "connected": True}

        except Exception as e:
            return {"error": str(e), "connected": False}

    async def get_upcoming_events(self, days: int = 7) -> dict:
        """Get upcoming calendar events."""
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            tokens = await self._get_credentials()
            if not tokens:
                return {"error": "Google Calendar not connected.", "connected": False}

            creds = Credentials(
                token=tokens.get("access_token"),
                refresh_token=tokens.get("refresh_token"),
                token_uri="https://oauth2.googleapis.com/token",
                client_id=settings.GOOGLE_CLIENT_ID,
                client_secret=settings.GOOGLE_CLIENT_SECRET
            )

            service = build("calendar", "v3", credentials=creds)
            now = datetime.utcnow().isoformat() + "Z"
            end = (datetime.utcnow() + timedelta(days=days)).isoformat() + "Z"

            events_result = service.events().list(
                calendarId="primary",
                timeMin=now,
                timeMax=end,
                maxResults=10,
                singleEvents=True,
                orderBy="startTime"
            ).execute()

            events = events_result.get("items", [])
            formatted = []
            for event in events:
                start = event["start"].get("dateTime", event["start"].get("date"))
                formatted.append({
                    "title": event.get("summary", "Untitled"),
                    "start": start,
                    "description": (event.get("description", "") or "")[:200],
                    "attendees": len(event.get("attendees", []))
                })

            return {"events": formatted, "days_ahead": days, "connected": True}

        except Exception as e:
            return {"error": str(e), "connected": False}

    @staticmethod
    def get_auth_url() -> str:
        """Generate Google OAuth URL."""
        from google_auth_oauthlib.flow import Flow
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": settings.GOOGLE_CLIENT_ID,
                    "client_secret": settings.GOOGLE_CLIENT_SECRET,
                    "redirect_uris": [settings.GOOGLE_REDIRECT_URI],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token"
                }
            },
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/calendar.readonly",
                "https://www.googleapis.com/auth/drive.readonly"
            ]
        )
        flow.redirect_uri = settings.GOOGLE_REDIRECT_URI
        url, _ = flow.authorization_url(prompt="consent", access_type="offline")
        return url
