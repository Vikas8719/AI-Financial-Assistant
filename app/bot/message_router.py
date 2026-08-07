"""
Message router — onboarding flow + AI agent dispatch.
"""
import logging
import re
from typing import List

from app.ai.agent import FinancialAgent
from app.ai.prompts import ONBOARDING_COMPLETE, ONBOARDING_START
from app.config import settings
from app.models.user_repo import get_user_profile_dict, update_user

logger = logging.getLogger("finbot.router")

STEP_KEY = "onboarding_step"


async def route_text_message(db, user, text: str) -> str:
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
    agent = FinancialAgent(db)
    return await agent.process_message(user.id, text, profile)


async def _run_onboarding(db, user, text: str) -> str:
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
            "_(e.g. 8:00 AM, 09:30 IST — or say skip to set it later)_"
        )

    if step == 4:
        briefing_time = None if _is_skip(text) else _parse_time(text)
        prefs[STEP_KEY] = 99
        await update_user(
            db, user.id,
            briefing_time=briefing_time,
            onboarded=True,
            preferences=prefs,
        )
        from app.models.user_repo import get_user
        user = await get_user(db, user.id)
        watchlist_val = user.watchlist if user else []
        interests_val = user.interests if user else []
        return ONBOARDING_COMPLETE.format(
            name=user.first_name if user else "there",
            role=user.role if user else "Finance Professional",
            watchlist=", ".join(list(watchlist_val)) if watchlist_val else "nothing yet",
            interests=", ".join(list(interests_val)) if interests_val else "general finance",
            briefing=briefing_time or "no daily brief set",
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


def _parse_time(text: str) -> str:
    match = re.search(r"\b(\d{1,2}:\d{2})\b", text)
    if match:
        return match.group(1)
    return text.strip()[:20]
