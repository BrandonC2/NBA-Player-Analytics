import argparse
import time
from datetime import datetime

import pandas as pd
import requests
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import players as nba_players

from .config import (
    THE_ODDS_API_KEY,
    ODDS_API_REGION,
    PRIZEPICKS_LINES_CSV,
    require_api_key,
)
from .utils_io import save_df, cache_path, load_csv_if_exists

# -------------------------------------------------
# Constants
# -------------------------------------------------
SPORT_KEY = "basketball_nba"
PLAYER_PROP_MARKETS = "player_points,player_rebounds,player_assists,player_threes"


# -------------------------------------------------
# NBA data fetchers (nba_api)
# -------------------------------------------------
def fetch_games_by_date(date_str: str, season: str) -> pd.DataFrame:
    """
    Pull games played on date_str within the given NBA season.
    date_str: "YYYY-MM-DD"
    season:   e.g., "2025-26"
    """
    lgf = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00"
    )
    games = lgf.get_data_frames()[0]
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"]).dt.date.astype(str)
    games_today = games[games["GAME_DATE"] == date_str]

    cols = [
        "GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "MATCHUP",
        "WL", "PTS", "REB", "AST", "PLUS_MINUS"
    ]
    return games_today[cols].drop_duplicates().reset_index(drop=True)


def fetch_players_lookup() -> pd.DataFrame:
    """
    Player name <-> id lookup from nba_api.
    """
    plist = nba_players.get_players()
    df = pd.DataFrame(plist)
    df["full_name_lower"] = df["full_name"].str.lower()
    return df


# -------------------------------------------------
# The Odds API — events + per-event player props
# -------------------------------------------------
def fetch_events(date_iso: str) -> list:
    """
    Get upcoming/live NBA events. We locally filter by date_iso if needed.
    """
    require_api_key("THE_ODDS_API_KEY", THE_ODDS_API_KEY)
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/events?regions={ODDS_API_REGION}&apiKey={THE_ODDS_API_KEY}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    events = r.json() or []

    # Optional: filter to events occurring on the given date (UTC date portion)
    # commence_time is ISO 8601; we compare by date prefix
    date_prefix = str(pd.to_datetime(date_iso).date())
    filtered = []
    for ev in events:
        ct = ev.get("commence_time")
        if not ct:
            continue
        # Normalize to date string
        ev_date = str(pd.to_datetime(ct).date())
        if ev_date == date_prefix:
            filtered.append(ev)
    return filtered


def fetch_event_player_props(event_id: str, markets: str = PLAYER_PROP_MARKETS) -> list:
    """
    Fetch player prop odds for a single event using the per-event endpoint.
    markets: comma-separated market keys, e.g. "player_points,player_rebounds,player_assists,player_threes"
    """
    require_api_key("THE_ODDS_API_KEY", THE_ODDS_API_KEY)
    url = (
        f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/events/{event_id}/odds"
        f"?regions={ODDS_API_REGION}&markets={markets}&oddsFormat=american&apiKey={THE_ODDS_API_KEY}"
    )
    r = requests.get(url, timeout=30)
    if r.status_code == 404:
        # Event can go stale; treat as empty
        return []
    r.raise_for_status()
    event = r.json() or {}
    rows = []

    ev_id = event.get("id")
    commence_time = event.get("commence_time")
    home_team = event.get("home_team")
    away_team = event.get("away_team")

    for bookmaker in event.get("bookmakers", []):
        book = bookmaker.get("title")
        for market in bookmaker.get("markets", []):
            market_key = market.get("key")  # e.g., player_points
            for outcome in market.get("outcomes", []):
                rows.append({
                    "event_id": ev_id,
                    "commence_time": commence_time,
                    "home_team": home_team,
                    "away_team": away_team,
                    "book": book,
                    "market": market_key,
                    "player": outcome.get("name"),
                    "american_odds": outcome.get("price"),
                    "line": outcome.get("point"),
                })
    return rows


def fetch_player_props_all_events(date_str: str) -> pd.DataFrame:
    """
    Pulls events for date_str, then fetches player props per event and concatenates.
    """
    events = fetch_events(date_str)
    if not events:
        return pd.DataFrame()

    all_rows = []
    for ev in events:
        ev_id = ev.get("id")
        if not ev_id:
            continue
        try:
            rows = fetch_event_player_props(ev_id, PLAYER_PROP_MARKETS)
            all_rows.extend(rows)
        except requests.HTTPError as e:
            print(f"[warn] event {ev_id} props fetch failed: {e}")
        # Be gentle with rate limits
        time.sleep(0.2)

    return pd.DataFrame(all_rows)


# -------------------------------------------------
# PrizePicks manual lines
# -------------------------------------------------
def load_prizepicks_lines_csv() -> pd.DataFrame:
    """
    Load your manual PrizePicks lines CSV; create a starter if missing.
    """
    df = load_csv_if_exists(PRIZEPICKS_LINES_CSV)
    if df is None:
        df = pd.DataFrame({
            "player": [],
            "team": [],
            "prop": [],
            "line": [],
            "direction": [],
            "source_date": [],
        })
        save_df(df, PRIZEPICKS_LINES_CSV)
    return df


# -------------------------------------------------
# Orchestration
# -------------------------------------------------
def fetch_foundation(date: str, season: str, save: bool = True):
    # 1) NBA games for date (from nba_api)
    try:
        games_df = fetch_games_by_date(date, season)
    except Exception as e:
        print("[warn] fetch_games_by_date failed:", e)
        games_df = pd.DataFrame()

    # 2) Player props odds via per-event endpoint (fixes 422)
    try:
        odds_df = fetch_player_props_all_events(date)
    except Exception as e:
        print("[warn] fetch_player_props_all_events failed:", e)
        odds_df = pd.DataFrame()

    # 3) PrizePicks manual lines
    pp_lines_df = load_prizepicks_lines_csv()

    if save:
        if not games_df.empty:
            save_df(games_df, cache_path("nba_games", date))
        if not odds_df.empty:
            save_df(odds_df, cache_path("odds_player_props", date))
        if not pp_lines_df.empty:
            save_df(pp_lines_df, cache_path("prizepicks_lines_copy", date))

    return {
        "games": games_df,
        "odds": odds_df,
        "prizepicks_lines": pp_lines_df,
    }


# -------------------------------------------------
# CLI entry
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Foundation fetch test")
    parser.add_argument("--date", type=str, default=datetime.utcnow().date().isoformat())
    parser.add_argument("--season", type=str, default="2025-26")
    parser.add_argument("--save", type=int, default=1)
    args = parser.parse_args()

    out = fetch_foundation(args.date, args.season, save=bool(args.save))

    print("\n=== NBA Games ===")
    print(out["games"].head())

    print("\n=== Odds (player props) ===")
    print(out["odds"].head())

    print("\n=== PrizePicks manual lines ===")
    print(out["prizepicks_lines"].head())
