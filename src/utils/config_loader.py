import os
from pathlib import Path
from dotenv import load_dotenv

def load_config() -> dict:
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path)

    return {
        "APIFY_API_TOKEN": os.getenv("APIFY_API_TOKEN"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ALPACA_API_KEY": os.getenv("ALPACA_API_KEY"),
        "ALPACA_SECRET_KEY": os.getenv("ALPACA_SECRET_KEY"),
        "ALPACA_BASE_URL": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    }

def load_watchlist() -> list:
    return [""
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "AMD",
        "INTC",
        "TSLA",
        "NFLX",
        "DIS",
        "BA",
        "LMT",
        "RTX",
        "NOC",
        "GD"
]