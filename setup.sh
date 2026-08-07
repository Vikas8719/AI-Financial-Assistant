#!/bin/bash
# FinBot — Quick Setup Script
set -e

echo "🤖 Setting up AI Financial Assistant..."

# 1. Python virtual environment
if [ ! -d "venv" ]; then
  echo "📦 Creating virtual environment..."
  python -m venv venv
fi

# 2. Activate & install deps
echo "📦 Installing dependencies..."
source venv/bin/activate || source venv/Scripts/activate 2>/dev/null
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 3. Environment file
if [ ! -f ".env" ]; then
  echo "⚙️  Creating .env from template..."
  cp .env.example .env
  echo "📝 Please fill in your API keys in .env before continuing."
  exit 1
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Fill in .env with your API keys"
echo "  2. Run:  uvicorn app.main:app --reload --port 8000"
echo "  3. Set Telegram webhook:"
echo "     curl -X POST 'https://api.telegram.org/bot<TOKEN>/setWebhook' \\"
echo "          -d 'url=https://your-domain.com/webhook'"
echo ""
echo "  Or with Docker:"
echo "     docker-compose up --build"
