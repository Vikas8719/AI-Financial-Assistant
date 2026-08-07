# 🤖 AI Financial Assistant — Telegram Bot

An AI-powered Financial Assistant that lives inside Telegram and helps finance professionals stay informed, conduct research, prepare for meetings, and make better decisions through natural conversations.

## 🏗️ Architecture

```
Telegram Bot (Text/Voice/Image)
        ↓
FastAPI Backend (Routes · Webhooks · Jobs)
        ↓
AI Layer (Groq LLM · pgvector memory)
    ↙       ↓       ↘        ↘
Finnhub  Yahoo    SEC      Web Search
         Finance  EDGAR    (Tavily)
        ↓
Document Intelligence (PDF · Annual Reports · SEC Filings)
        ↓
PostgreSQL (Neon · pgvector)
```

## 🚀 Tech Stack

- **Backend:** FastAPI (Python)
- **Database:** PostgreSQL with pgvector (Neon)
- **AI:** Groq LLM (llama-3.3-70b-versatile)
- **Memory:** pgvector semantic search
- **Telegram:** python-telegram-bot
- **Scheduler:** APScheduler
- **Financial Data:** Finnhub, Yahoo Finance, SEC EDGAR
- **Web Search:** Tavily / DuckDuckGo
- **Document AI:** PyMuPDF, pdfplumber

## 📁 Project Structure

```
ai-financial-assistant/
├── app/
│   ├── main.py                    # FastAPI entry point
│   ├── config.py                  # Settings & env vars
│   ├── database.py                # DB connection & models
│   │
│   ├── bot/
│   │   ├── telegram_handler.py    # Telegram webhook handler
│   │   ├── message_router.py      # Routes messages to handlers
│   │   └── voice_handler.py       # Voice message processing
│   │
│   ├── ai/
│   │   ├── agent.py               # Main AI agent (Groq LLM)
│   │   ├── memory.py              # pgvector conversation memory
│   │   ├── tools.py               # AI tool definitions
│   │   └── prompts.py             # System prompts
│   │
│   ├── services/
│   │   ├── finnhub_service.py     # Live prices & news
│   │   ├── yahoo_finance.py       # Fundamentals & charts
│   │   ├── sec_edgar.py           # SEC filings
│   │   ├── web_search.py          # DuckDuckGo / Tavily
│   │   └── document_service.py    # PDF & document analysis
│   │
│   ├── scheduler/
│   │   ├── jobs.py                # APScheduler jobs
│   │   └── alerts.py             # Price & news alerts
│   │
│   ├── api/
│   │   ├── routes.py              # REST API routes
│   │   └── auth.py                # OAuth handlers (Google)
│   │
│   └── models/
│       ├── user.py                # User model
│       ├── conversation.py        # Conversation model
│       └── alert.py               # Alert model
│
├── migrations/                    # Alembic migrations
├── tests/                         # Unit & integration tests
├── .env.example                   # Environment variables template
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker container
├── docker-compose.yml             # Full stack compose
└── README.md
```

## ⚙️ Setup

### 1. Clone & Install

```bash
git clone <repo>
cd ai-financial-assistant
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables

```bash
cp .env.example .env
# Fill in your API keys in .env
```

### 3. Database Setup

```bash
# Run migrations
alembic upgrade head
```

### 4. Run the Bot

```bash
# Development
uvicorn app.main:app --reload --port 8000

# Set Telegram webhook
curl -X POST "https://api.telegram.org/bot{TOKEN}/setWebhook" \
  -d "url=https://your-domain.com/webhook"
```

> The webhook endpoint is exposed at `/webhook` by the FastAPI app.

> Google OAuth is available under `/auth/google` and the callback endpoint is `/auth/google/callback`.

## 🔑 Required API Keys

| Service              | Purpose            | Get It           |
| -------------------- | ------------------ | ---------------- |
| `TELEGRAM_BOT_TOKEN` | Bot access         | @BotFather       |
| `GROQ_API_KEY`       | LLM inference      | console.groq.com |
| `FINNHUB_API_KEY`    | Live prices & news | finnhub.io       |
| `TAVILY_API_KEY`     | Web search         | tavily.com       |
| `DATABASE_URL`       | PostgreSQL         | neon.tech        |

## 📌 Key Features

- ✅ Natural conversational onboarding
- ✅ Morning market brief (scheduled)
- ✅ Live stock prices & news via Finnhub
- ✅ Company research via Yahoo Finance
- ✅ SEC EDGAR filings search
- ✅ PDF & document intelligence
- ✅ Voice message support
- ✅ Personalized watchlist & alerts
- ✅ pgvector-powered conversation memory
- ✅ Google Calendar/Gmail integration (OAuth)
