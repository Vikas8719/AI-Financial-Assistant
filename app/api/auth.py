"""
Google OAuth 2.0 flow.
GET  /auth/google?user_id=<telegram_id>   → redirects to Google consent
GET  /auth/google/callback                → exchanges code, saves tokens
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.user_repo import get_user, update_user

logger = logging.getLogger("finbot.auth")
router = APIRouter()

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

SUCCESS_HTML = """
<!DOCTYPE html><html><head><title>FinBot — Connected!</title>
<style>
  body{font-family:system-ui,sans-serif;display:flex;align-items:center;
       justify-content:center;height:100vh;margin:0;background:#f0fdf4;}
  .card{text-align:center;padding:2rem;background:#fff;border-radius:1rem;
        box-shadow:0 4px 24px rgba(0,0,0,.08);max-width:380px;}
  h2{color:#16a34a;margin-bottom:.5rem;}
  p{color:#374151;margin:.5rem 0;}
  .emoji{font-size:3rem;margin-bottom:1rem;}
</style></head><body>
<div class="card">
  <div class="emoji">✅</div>
  <h2>Google Connected!</h2>
  <p>Your account has been linked to FinBot.</p>
  <p>Return to Telegram and keep chatting 🤖</p>
</div>
</body></html>
"""

ERROR_HTML = """
<!DOCTYPE html><html><head><title>FinBot — Error</title>
<style>
  body{{font-family:system-ui,sans-serif;display:flex;align-items:center;
       justify-content:center;height:100vh;margin:0;background:#fef2f2;}}
  .card{{text-align:center;padding:2rem;background:#fff;border-radius:1rem;
        box-shadow:0 4px 24px rgba(0,0,0,.08);max-width:380px;}}
  h2{{color:#dc2626;}}
  p{{color:#374151;}}
</style></head><body>
<div class="card">
  <h2>⚠️ Something went wrong</h2>
  <p>{error}</p>
  <p>Please try again from Telegram.</p>
</div>
</body></html>
"""


def _build_flow():
    """Create Google OAuth flow — returns None if credentials not configured."""
    if not all([settings.GOOGLE_CLIENT_ID, settings.GOOGLE_CLIENT_SECRET, settings.GOOGLE_REDIRECT_URI]):
        return None
    from google_auth_oauthlib.flow import Flow
    return Flow.from_client_config(
        {
            "web": {
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [settings.GOOGLE_REDIRECT_URI],
            }
        },
        scopes=SCOPES,
        redirect_uri=settings.GOOGLE_REDIRECT_URI,
    )


@router.get("/google", tags=["OAuth"])
async def google_oauth_start(user_id: int):
    """Initiate Google OAuth — redirect user to Google consent screen."""
    flow = _build_flow()
    if not flow:
        raise HTTPException(status_code=400, detail="Google OAuth is not configured on this server.")

    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        state=str(user_id),
        prompt="consent",
    )
    return RedirectResponse(auth_url)


@router.get("/google/callback", tags=["OAuth"])
async def google_oauth_callback(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Handle Google OAuth callback — exchange code for tokens and save."""
    state = request.query_params.get("state")
    if not state or not state.isdigit():
        return HTMLResponse(ERROR_HTML.format(error="Invalid OAuth state parameter."), status_code=400)

    user_id = int(state)
    user = await get_user(db, user_id)
    if not user:
        return HTMLResponse(ERROR_HTML.format(error="User not found. Please restart the bot."), status_code=404)

    flow = _build_flow()
    if not flow:
        return HTMLResponse(ERROR_HTML.format(error="OAuth not configured."), status_code=500)

    try:
        # Allow http in development
        import os
        os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1" if settings.DEBUG else "0")

        flow.fetch_token(authorization_response=str(request.url))
        creds = flow.credentials

        token_data = {
            "access_token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": list(creds.scopes or []),
        }

        await update_user(db, user_id, google_tokens=token_data)
        logger.info(f"✅ Google account linked for user {user_id}")

        # Notify user in Telegram
        try:
            from telegram import Bot
            tg = Bot(token=settings.TELEGRAM_BOT_TOKEN)
            await tg.send_message(
                chat_id=user_id,
                text=(
                    "✅ *Google account connected successfully!*\n\n"
                    "I now have access to your Gmail and Calendar. "
                    "Try asking: _'Summarize my emails about Tesla'_ or "
                    "_'What meetings do I have this week?'_"
                ),
                parse_mode="Markdown"
            )
        except Exception:
            pass

        return HTMLResponse(SUCCESS_HTML)

    except Exception as e:
        logger.exception(f"OAuth callback failed for user {user_id}: {e}")
        return HTMLResponse(ERROR_HTML.format(error="Authentication failed. Please try again."), status_code=500)
