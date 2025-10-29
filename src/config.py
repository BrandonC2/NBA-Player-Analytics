import os
from dotenv import load_dotenv

# load .env if present
load_dotenv()

# odds api
THE_ODDS_API_KEY_ENV = "THE_ODDS_API_KEY"
THE_ODDS_API_KEY = os.getenv(THE_ODDS_API_KEY_ENV, "")
ODDS_API_REGION = os.getenv("ODDS_API_REGION", "us")
ODDS_API_MARKETS = os.getenv("ODDS_API_MARKETS", "player_props")

# caching
CACHE_DIR = os.getenv("CACHE_DIR", "data/cache")
CACHE_TTL_MINUTES = int(os.getenv("CACHE_TTL_MINUTES", "30"))

def require_api_key(name: str, value: str):
    if not value:
        raise RuntimeError(f"Missing API key for {name}. Set {name} in your .env file.")

# manual lines path
PRIZEPICKS_LINES_CSV = os.path.join("data", "prizepicks_lines.csv")

# ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)
