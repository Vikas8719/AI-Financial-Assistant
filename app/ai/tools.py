"""Tool definitions for the AI agent — Groq function calling."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get real-time stock price, change, and basic info for a ticker symbol. Use company name or ticker — both work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker or company name e.g. AAPL, TSLA, Tesla, Microsoft, RELIANCE.NS",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_news",
            "description": "Get latest news articles for a company or stock symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker or company name"},
                    "days": {
                        "type": "integer",
                        "description": "Number of days back to fetch news (default 7)",
                        "default": 7,
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_fundamentals",
            "description": "Get company financials: revenue, earnings, P/E ratio, market cap, profit margins. Use ticker or company name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker or company name e.g. MSFT, Google, Apple"}
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_sec_filings",
            "description": "Search SEC EDGAR for company filings like 10-K, 10-Q, 8-K.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "Company name to search",
                    },
                    "filing_type": {
                        "type": "string",
                        "description": "Filing type: 10-K, 10-Q, 8-K, DEF 14A",
                        "default": "10-K",
                    },
                },
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for latest financial news, company information, or market events. Use when other tools fail or for very recent news.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_overview",
            "description": "Get current market overview: major indices (S&P 500, NASDAQ, Dow Jones, Nifty 50), sector performance.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_earnings_calendar",
            "description": "Get upcoming earnings announcements for the next N days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Days ahead to look (default 7)",
                        "default": 7,
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_companies",
            "description": (
                "Compare two or more companies side by side on key financial metrics: "
                "revenue, market cap, P/E, margins, growth, debt etc. "
                "ALWAYS use this for queries like 'compare Tesla and Google', "
                "'Tesla vs Microsoft', 'which is better AAPL or MSFT'. "
                "Pass company names OR tickers — both work. "
                "Has automatic fallback: Yahoo Finance → Finnhub → Web Search, "
                "so it NEVER fails even if one data source is unavailable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol1": {
                        "type": "string",
                        "description": "First company name or ticker e.g. 'Tesla' or 'TSLA'",
                    },
                    "symbol2": {
                        "type": "string",
                        "description": "Second company name or ticker e.g. 'Google' or 'GOOGL'",
                    },
                },
                "required": ["symbol1", "symbol2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_document",
            "description": "Analyze a previously uploaded financial document and answer questions about it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "integer",
                        "description": "Optional document ID. Omit to use the most recently uploaded document.",
                    },
                    "question": {
                        "type": "string",
                        "description": "Question to answer about the document",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_alert",
            "description": "Set a price or news alert for a stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"},
                    "alert_type": {
                        "type": "string",
                        "description": "Type: price_above, price_below, pct_change, news",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Price or percentage threshold",
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description of the alert",
                    },
                },
                "required": ["symbol", "alert_type", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_gmail_summary",
            "description": "Search and summarize Gmail emails related to a company or topic (requires Google auth).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query e.g. company name or topic",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_calendar_events",
            "description": "Get upcoming calendar events for meeting prep (requires Google auth).",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Days ahead to check",
                        "default": 7,
                    }
                },
            },
        },
    },
]
