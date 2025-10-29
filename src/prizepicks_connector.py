"""
Live PrizePicks line connector (rate-limit tolerant) + tag-based side locks.

Adds columns:
- pp_tags: concatenated text we scanned from the PrizePicks payload (debuggable)
- side_lock: "OVER" for Demon/Goblin specials, else ""

Usage:
  python -m src.prizepicks_connector --league NBA --out data/prizepicks_live.csv
"""

import argparse
import random
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import requests

# ----------------------------------------
# Known league ids (fallbacks)
# ----------------------------------------
KNOWN_LEAGUE_IDS = {
    "NBA": 7,
    "WNBA": 3,
    "NFL": 9,
    "NHL": 8,
    "MLB": 2,
}

PP_BASE = "https://api.prizepicks.com"

DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.prizepicks.com",
    "Referer": "https://www.prizepicks.com/",
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
}

STAT_TO_PROP = {
    "points": "PTS",
    "rebounds": "REB",
    "assists": "AST",
    "three_points_made": "3PM",
    "three_pointers_made": "3PM",
    "threes": "3PM",
}

# Demon/Goblin → OVER-only
_DEMON_GOBLIN_RE = re.compile(r"\b(?:red|green)?\s*(?:demon|goblin)s?\b", re.IGNORECASE)

def _flatten_strings(x: Union[dict, list, str, int, float, None]) -> List[str]:
    """Recursively collect ALL strings from nested dict/list structures."""
    out = []
    if isinstance(x, str):
        s = x.strip()
        if s:
            out.append(s)
    elif isinstance(x, dict):
        for v in x.values():
            out.extend(_flatten_strings(v))
    elif isinstance(x, list):
        for v in x:
            out.extend(_flatten_strings(v))
    # ignore numbers/None
    return out

def _detect_pp_tags(attrs: dict, item: dict) -> str:
    """
    Join EVERY string from attributes + selected top-level item fields.
    This is robust to schema changes (promo_name, badge_text, marketing_name, etc.).
    """
    texts = _flatten_strings(attrs) + _flatten_strings({
        "type": item.get("type"),
        "id": item.get("id"),
        "relationships": item.get("relationships"),
    })
    # Dedup short strings to keep size reasonable
    uniq = []
    seen = set()
    for t in texts:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return " | ".join(uniq[:200])  # cap length defensively

def _infer_side_lock_from_tags(pp_tags: str) -> str:
    if pp_tags and _DEMON_GOBLIN_RE.search(pp_tags):
        return "OVER"
    return ""

# ----------------------------------------
# HTTP helpers (with backoff)
# ----------------------------------------
def _sleep_with_jitter(base: float, factor: float, attempt: int):
    delay = base * (factor ** attempt) + random.uniform(0, 0.25)
    time.sleep(delay)

def _get_json(url: str, headers: Dict[str, str], tries: int = 5, base_sleep: float = 0.6, factor: float = 1.8) -> Any:
    last_exc = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    try:
                        time.sleep(float(retry_after))
                    except Exception:
                        _sleep_with_jitter(base_sleep, factor, i)
                else:
                    _sleep_with_jitter(base_sleep, factor, i)
                continue
            if 500 <= r.status_code < 600:
                _sleep_with_jitter(base_sleep, factor, i)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            _sleep_with_jitter(base_sleep, factor, i)
    if last_exc:
        raise last_exc

def _try_fetch_leagues(headers: Dict[str, str], tries: int, base_sleep: float) -> pd.DataFrame:
    url = f"{PP_BASE}/leagues"
    try:
        js = _get_json(url, headers, tries=tries, base_sleep=base_sleep)
    except Exception:
        return pd.DataFrame()
    if not js or "data" not in js:
        return pd.DataFrame()
    rows = []
    for item in js["data"]:
        lid = item.get("id")
        attrs = (item.get("attributes") or {})
        rows.append({
            "id": int(lid) if lid and str(lid).isdigit() else lid,
            "name": attrs.get("name") or attrs.get("abbreviation") or attrs.get("icon"),
            "active": attrs.get("active", True),
        })
    return pd.DataFrame(rows)

def _resolve_league_id(league_name: str, headers: Dict[str, str], tries: int, base_sleep: float, override_id: int = None) -> int:
    if override_id:
        return int(override_id)
    leagues = _try_fetch_leagues(headers, tries, base_sleep)
    if not leagues.empty:
        mask = leagues["name"].str.upper() == league_name.upper()
        found = leagues[mask]
        if not found.empty:
            return int(found.iloc[0]["id"])
    if league_name.upper() in KNOWN_LEAGUE_IDS:
        return KNOWN_LEAGUE_IDS[league_name.upper()]
    raise RuntimeError(f"Unable to resolve league id for '{league_name}'. Pass --league_id manually.")

# ----------------------------------------
# Parsing helpers
# ----------------------------------------
def _index_included(included: List[dict]) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    players_by_id, teams_by_id = {}, {}
    for node in included or []:
        typ = node.get("type")
        nid = str(node.get("id"))
        attrs = node.get("attributes") or {}
        if typ in ("new_player", "player"):
            players_by_id[nid] = {
                "name": attrs.get("name") or attrs.get("display_name") or attrs.get("full_name"),
                "team": (attrs.get("team") or attrs.get("team_abbreviation") or attrs.get("team_name") or ""),
            }
        elif typ in ("team", "new_team"):
            abbr = attrs.get("abbr") or attrs.get("abbreviation") or attrs.get("name") or ""
            teams_by_id[nid] = {"abbr": abbr, "name": attrs.get("name") or abbr}
    return players_by_id, teams_by_id

def _pp_stat_to_prop(stat: str) -> str:
    stat = (stat or "").strip().lower()
    return STAT_TO_PROP.get(stat, stat.upper())

# ----------------------------------------
# Core fetch (paginated)
# ----------------------------------------
def fetch_prizepicks_projections(
    league_name: str = "NBA",
    league_id: int = None,
    per_page: int = 300,
    pages: int = 3,
    base_sleep: float = 0.6,
    max_tries: int = 6,
) -> pd.DataFrame:
    headers = DEFAULT_HEADERS.copy()
    lid = _resolve_league_id(league_name, headers, tries=max_tries, base_sleep=base_sleep, override_id=league_id)

    all_rows = []
    now_date = datetime.now(timezone.utc).date().isoformat()

    for page in range(1, pages + 1):
        url = (
            f"{PP_BASE}/projections?"
            f"league_id={lid}&per_page={per_page}&page={page}&single_stat=true&game_mode=pickem"
        )
        js = _get_json(url, headers, tries=max_tries, base_sleep=base_sleep)
        if not js or "data" not in js or not js["data"]:
            break

        players_by_id, teams_by_id = _index_included(js.get("included", []))

        for item in js["data"]:
            attrs = item.get("attributes") or {}
            stat = attrs.get("stat_type") or attrs.get("stat") or ""
            prop = _pp_stat_to_prop(stat)
            line = attrs.get("line_score")
            league_id_seen = attrs.get("league_id") or lid

            # player
            new_player_id = (
                attrs.get("new_player_id") or
                (item.get("relationships", {}).get("new_player", {}).get("data") or {}).get("id")
            )
            player_name, team_abbr = None, ""
            if new_player_id and str(new_player_id) in players_by_id:
                p = players_by_id[str(new_player_id)]
                player_name = p.get("name")
                team_abbr = (p.get("team") or "").upper()
            else:
                player_name = attrs.get("description")
                if player_name and " - " in player_name:
                    player_name = player_name.split(" - ", 1)[0].strip()

            if not player_name:
                continue
            try:
                line_val = float(line)
            except Exception:
                continue

            # Tag detection (deep scan) & side lock inference
            pp_tags = _detect_pp_tags(attrs, item)
            side_lock = _infer_side_lock_from_tags(pp_tags)

            all_rows.append({
                "player": player_name,
                "team": team_abbr,
                "prop": prop,
                "line": line_val,
                "source_date": now_date,
                "prizepicks_stat": stat,
                "league_id": int(league_id_seen) if str(league_id_seen).isdigit() else league_id_seen,
                "pp_tags": pp_tags,
                "side_lock": side_lock,   # "OVER" for Demon/Goblin, "" otherwise
            })

        time.sleep(base_sleep)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["player", "team", "prop", "line"]).reset_index(drop=True)
    return df

# ----------------------------------------
# CLI
# ----------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fetch live PrizePicks projections and write CSV")
    ap.add_argument("--league", type=str, default="NBA")
    ap.add_argument("--league_id", type=int, default=None, help="Override league id if known")
    ap.add_argument("--per_page", type=int, default=300, help="Items per page (lower if you see 429)")
    ap.add_argument("--pages", type=int, default=3, help="Max pages to fetch (lower if you see 429)")
    ap.add_argument("--sleep", type=float, default=0.6, help="Base sleep/backoff seconds")
    ap.add_argument("--max_tries", type=int, default=6, help="Max tries per HTTP request")
    ap.add_argument("--out", type=str, default="data/prizepicks_live.csv")
    args = ap.parse_args()

    df = fetch_prizepicks_projections(
        league_name=args.league,
        league_id=args.league_id,
        per_page=args.per_page,
        pages=args.pages,
        base_sleep=args.sleep,
        max_tries=args.max_tries,
    )
    if df.empty:
        print("[info] No PrizePicks data fetched. Reasons may include: rate limit, slate not posted yet, or endpoint changes.")
    else:
        # Debug summary
        is_locked = (df["side_lock"].astype(str).str.upper() == "OVER").sum()
        print(f"[ok] Fetched {len(df)} rows | Demon/Goblin OVER locks detected: {is_locked}")
        df.to_csv(args.out, index=False)
        print(f"[ok] Wrote → {args.out}")
        print(df.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
