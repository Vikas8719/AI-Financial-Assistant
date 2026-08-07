"""User repository — CRUD operations for users and alerts."""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import Optional
from datetime import datetime

from app.database import User, Alert


# ─── User Operations ──────────────────────────────────────────────────────────

async def get_user(db: AsyncSession, user_id: int) -> Optional[User]:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def get_or_create_user(db: AsyncSession, user_id: int, **kwargs) -> tuple[User, bool]:
    """Get existing user or create new one. Returns (user, created)."""
    user = await get_user(db, user_id)
    if user:
        return user, False

    user = User(id=user_id, **kwargs)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user, True


async def update_user(db: AsyncSession, user_id: int, **fields) -> Optional[User]:
    """Update user fields."""
    fields["updated_at"] = datetime.utcnow()
    await db.execute(
        update(User).where(User.id == user_id).values(**fields)
    )
    await db.commit()
    return await get_user(db, user_id)


async def add_to_watchlist(db: AsyncSession, user_id: int, symbol: str) -> list:
    """Add symbol to user's watchlist."""
    user = await get_user(db, user_id)
    if not user:
        return []
    watchlist = list(user.watchlist or [])
    symbol = symbol.upper()
    if symbol not in watchlist:
        watchlist.append(symbol)
    await update_user(db, user_id, watchlist=watchlist)
    return watchlist


async def remove_from_watchlist(db: AsyncSession, user_id: int, symbol: str) -> list:
    """Remove symbol from user's watchlist."""
    user = await get_user(db, user_id)
    if not user:
        return []
    watchlist = [s for s in (user.watchlist or []) if s.upper() != symbol.upper()]
    await update_user(db, user_id, watchlist=watchlist)
    return watchlist


async def get_user_profile_dict(db: AsyncSession, user_id: int) -> dict:
    """Get user profile as dict for AI context."""
    user = await get_user(db, user_id)
    if not user:
        return {}
    return {
        "user_id": user_id,
        "name": user.first_name or "User",
        "role": user.role,
        "watchlist": user.watchlist or [],
        "interests": user.interests or [],
        "briefing_time": user.briefing_time,
        "timezone": user.timezone,
        "onboarded": user.onboarded,
        "google_connected": bool(user.google_tokens),
        "preferences": user.preferences or {}
    }


# ─── Alert Operations ─────────────────────────────────────────────────────────

async def create_alert(db: AsyncSession, user_id: int, alert_data: dict) -> Alert:
    """Create a new price/news alert."""
    alert = Alert(
        user_id=user_id,
        alert_type=alert_data.get("alert_type", "news"),
        symbol=alert_data.get("symbol", "").upper() if alert_data.get("symbol") else None,
        condition=alert_data.get("condition"),
        threshold=alert_data.get("threshold"),
        description=alert_data.get("description", ""),
        active=True
    )
    db.add(alert)
    await db.commit()
    await db.refresh(alert)
    return alert


async def get_active_alerts(db: AsyncSession, user_id: int) -> list[Alert]:
    """Get all active alerts for a user."""
    result = await db.execute(
        select(Alert).where(Alert.user_id == user_id, Alert.active == True)
    )
    return result.scalars().all()


async def get_all_price_alerts(db: AsyncSession) -> list[Alert]:
    """Get all active price alerts across all users (for scheduler)."""
    result = await db.execute(
        select(Alert).where(
            Alert.active == True,
            Alert.alert_type.in_(["price_above", "price_below", "pct_change"])
        )
    )
    return result.scalars().all()


async def get_users_with_briefing(db: AsyncSession, current_hour: str) -> list[User]:
    """Get users who want a briefing at this hour."""
    result = await db.execute(
        select(User).where(
            User.briefing_time == current_hour,
            User.onboarded == True
        )
    )
    return result.scalars().all()
