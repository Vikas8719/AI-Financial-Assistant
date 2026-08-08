"""
Message router — onboarding flow + AI agent dispatch.
Auto-injects latest uploaded document context into every AI call.
"""
import logging
import re
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.agent import FinancialAgent
from app.ai.prompts import ONBOARDING_COMPLETE, ONBOARDING_START
from app.config import settings
from app.models.user_repo import get_user_profile_dict, update_user

logger = logging.getLogger("finbot.router")

STEP_KEY = "onboarding_step"


async def _get_latest_document_context(db: AsyncSession, user_id: int) -> Optional[str]:
    """Fetch the most recently uploaded document's content for context injection."""
    from app.database import Document
    result = await db.execute(
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(Document.created_at.desc())
        .limit(1)
    )
    doc = result.scalar_one_or_none()
    if not doc or not doc.content or len(doc.content.strip()) < 50:
        return None

    context_parts = [f"📄 Uploaded Document: {doc.filename}"]
    if doc.summary:
        context_parts.append(f"Summary: {doc.summary}")
    if doc.content:
        context_parts.append(f"Full Content:\n{doc.content[:8000]}")
    return "\n\n".join(context_parts)


async def route_text_message(db: AsyncSession, user, text: str) -> str:
    text = text.strip()

    if text.lower() in ("/connect", "connect google", "link google"):
        auth_url = f"{settings.WEBHOOK_URL.rstrip('/')}/auth/google?user_id={user.id}"
        return (
            f"🔗 Connect your Google account to unlock Gmail & Calendar features:\n\n"
            f"{auth_url}\n\n"
            "After connecting, come back and keep chatting!"
        )

    if not user.onboarded:
        return await _run_onboarding(db, user, text)

    profile = await get_user_profile_dict(db, user.id)
    document_context = await _get_latest_document_context(db, user.id)

    agent = FinancialAgent(db)
    return await agent.process_message(user.id, text, profile, document_context=document_context)


async def _run_onboarding(db: AsyncSession, user, text: str) -> str:
    prefs = dict(user.preferences or {})
    step = int(prefs.get(STEP_KEY, 0))

    if step == 0:
        prefs[STEP_KEY] = 1
        await update_user(db, user.id, preferences=prefs)
        return ONBOARDING_START.format(name=user.first_name or "there")

    if step == 1:
        role = text if not _is_skip(text) else "Finance Professional"
        prefs[STEP_KEY] = 2
        await update_user(db, user.id, role=role, preferences=prefs)
        return (
            "Got it! 📋\n\n"
            "Which *companies, stocks, or sectors* are you actively following?\n"
            "_(e.g. AAPL, TSLA, AI sector, Indian banks — or say skip)_"
        )

    if step == 2:
        watchlist: List[str] = [] if _is_skip(text) else _parse_symbols(text)
        prefs[STEP_KEY] = 3
        await update_user(db, user.id, watchlist=watchlist, preferences=prefs)
        wl_str = ", ".join(watchlist) if watchlist else "nothing yet"
        return (
            f"Perfect — I'll keep an eye on *{wl_str}* for you. 👀\n\n"
            "What type of *financial insights* matter most to you?\n"
            "_(e.g. earnings, SEC filings, analyst ratings, macro news — or say skip)_"
        )

    if step == 3:
        interests: List[str] = [] if _is_skip(text) else _parse_list(text)
        prefs[STEP_KEY] = 4
        await update_user(db, user.id, interests=interests, preferences=prefs)
        return (
            "Noted! 🎯\n\n"
            "When would you like your *daily market briefing*?\n"
            "_(e.g. 8:00 AM, 09:30 — or say skip)_\n\n"
            "Also, what's your timezone?\n"
            "_(e.g. Asia/Kolkata, America/New_York, Europe/London — or say IST/EST/GMT)_"
        )

    if step == 4:
        # Parse both briefing time and timezone from same message
        briefing_time = None if _is_skip(text) else _parse_time(text)
        timezone = _parse_timezone(text)
        prefs[STEP_KEY] = 99
        await update_user(
            db, user.id,
            briefing_time=briefing_time,
            timezone=timezone,
            onboarded=True,
            preferences=prefs,
        )
        from app.models.user_repo import get_user
        user = await get_user(db, user.id)
        watchlist_val = list(user.watchlist or [])
        interests_val = list(user.interests or [])
        return ONBOARDING_COMPLETE.format(
            name=user.first_name or "there",
            role=user.role or "Finance Professional",
            watchlist=", ".join(watchlist_val) if watchlist_val else "nothing yet",
            interests=", ".join(interests_val) if interests_val else "general finance",
            briefing=f"{briefing_time} ({timezone})" if briefing_time else "no daily brief set",
        )

    prefs[STEP_KEY] = 0
    await update_user(db, user.id, preferences=prefs)
    return ONBOARDING_START.format(name=user.first_name or "there")


def _is_skip(text: str) -> bool:
    return text.lower().strip() in {"skip", "s", "no", "none", "nope", "-"}


def _parse_symbols(text: str) -> List[str]:
    candidates = re.split(r"[,\s/|&]+", text)
    result = []
    for item in candidates:
        item = item.strip().strip(".")
        if not item or _is_skip(item):
            continue
        upper = item.upper()
        if re.match(r"^[A-Z0-9.\-]{1,12}$", upper):
            result.append(upper)
        elif len(item) > 2:
            result.append(item.title())
    return list(dict.fromkeys(result))[:15]


def _parse_list(text: str) -> List[str]:
    parts = re.split(r"[,\n;]+", text)
    return [p.strip() for p in parts if p.strip() and not _is_skip(p.strip())]


def _parse_time(text: str) -> Optional[str]:
    match = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if match:
        return f"{int(match.group(1)):02d}:{match.group(2)}"
    # Handle "8 AM", "8AM"
    match = re.search(r"\b(\d{1,2})\s*(am|pm)\b", text.lower())
    if match:
        hour = int(match.group(1))
        if match.group(2) == "pm" and hour != 12:
            hour += 12
        elif match.group(2) == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:00"
    return text.strip()[:20] if not _is_skip(text) else None


def _parse_timezone(text: str) -> str:
    """Extract timezone from user message."""
    text_lower = text.lower()

    # Common shorthand mappings
    tz_map = {
        "ist": "Asia/Kolkata",
        "india": "Asia/Kolkata",
        "kolkata": "Asia/Kolkata",
        "mumbai": "Asia/Kolkata",
        "est": "America/New_York",
        "edt": "America/New_York",
        "new york": "America/New_York",
        "pst": "America/Los_Angeles",
        "pdt": "America/Los_Angeles",
        "gmt": "Europe/London",
        "utc": "UTC",
        "cst": "America/Chicago",
        "mst": "America/Denver",
        "dubai": "Asia/Dubai",
        "uae": "Asia/Dubai",
        "singapore": "Asia/Singapore",
        "sgt": "Asia/Singapore",
        "jst": "Asia/Tokyo",
        "japan": "Asia/Tokyo",
        "london": "Europe/London",
        "paris": "Europe/Paris",
        "sydney": "Australia/Sydney",
        "aest": "Australia/Sydney",
    }

    for key, tz in tz_map.items():
        if key in text_lower:
            return tz

    # Try full timezone name like "Asia/Kolkata"
    import pytz
    tz_match = re.search(r"[A-Z][a-z]+/[A-Z][a-z_]+", text)
    if tz_match:
        tz_str = tz_match.group(0)
        if tz_str in pytz.all_timezones:
            return tz_str

    # Default to IST for Indian users
    return "Asia/Kolkata"
