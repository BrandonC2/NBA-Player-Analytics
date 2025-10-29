"""
Rolling metrics utilities using nba_api player game logs + rich prop support.

Supported canonical props (case-insensitive):
  Atomic:  PTS, REB, AST, 3PM(FG3M), 3PA(FG3A), FGA, FGM, STL, BLK, TOV, FTA, 2PA
  Combos:  PR(PTS+REB), PA(PTS+AST), RA(REB+AST), PRA(PTS+REB+AST), STOCKS(BLK+STL)

Common aliases auto-normalize upstream in build_board_all.py (e.g., "3-PT MADE" -> 3PM,
"PTS+REBS+ASTS" -> PRA, "BLKS+STLS" -> STOCKS).
"""

from typing import Optional, Tuple
import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players as nba_players
import re
import unicodedata
from functools import lru_cache
from typing import Optional

# ----------------------------------------
# Player lookup (robust, with alias overrides)
# ----------------------------------------

_SUFFIX_RE = re.compile(r"\b(jr\.?|sr\.?|ii|iii|iv|v)\b", flags=re.IGNORECASE)

# Common first-name variants and known PP↔NBA differences
ALIAS_OVERRIDES = {
    # normalized -> normalized target full name as it appears in nba_api
    "nicolas claxton": "nic claxton",
    "carlton carrington": "bub carrington",
    # add more as needed:
    # "michael porter jr": "michael porter jr",     # example of identical (here for pattern)
    # "al farouq aminu": "al-farouq aminu",
    # "kevin porter jr": "kevin porter jr",
    # "kristaps porzingis": "kristaps porziņģis",   # accents handled below anyway
}

FIRSTNAME_VARIANTS = {
    "nicolas": "nic",
    "nicholas": "nick",
    "michael": "mike",
    "steven": "steve",
    "william": "bill",
    "robert": "bob",
    "james": "jim",
    "james": "jimmy",
    "joseph": "joe",
    "jonathan": "jon",
    "nicholas": "nico",
}

def _normalize_name(s: str) -> str:
    # lower, remove accents, drop punctuation & suffixes, collapse spaces
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _SUFFIX_RE.sub("", s)
    s = re.sub(r"[^a-z\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@lru_cache(maxsize=1)
def _players_index_df():
    plist = nba_players.get_players()  # [{id, full_name, is_active}, ...]
    df = pd.DataFrame(plist)
    df["full_name"] = df["full_name"].astype(str)
    df["norm_name"] = df["full_name"].apply(_normalize_name)
    toks = df["norm_name"].str.split(" ", expand=True).fillna("")
    df["first"] = toks[0]
    df["last"]  = toks.iloc[:, -1]
    # best-effort team tokens sometimes included in names list? (usually not)
    return df

def _apply_firstname_variant(norm: str) -> str:
    parts = norm.split()
    if not parts:
        return norm
    first = parts[0]
    rest = " ".join(parts[1:])
    alt = FIRSTNAME_VARIANTS.get(first, first)
    return (alt + " " + rest).strip()

def get_player_id_by_name(name: str, team_abbr: Optional[str] = None) -> Optional[int]:
    """
    Robust resolver:
      0) alias overrides
      1) exact normalized match
      2) startswith on normalized
      3) contains (first & last tokens)
      4) first-name variant (Nicolas->Nic, etc.), redo steps 1–3
      5) unique last-name fallback (prefer active)
    """
    df = _players_index_df()
    norm = _normalize_name(name)

    # 0) alias override
    alias = ALIAS_OVERRIDES.get(norm)
    if alias:
        norm = _normalize_name(alias)

    # helper to choose one row (prefer active)
    def choose_one(cand: pd.DataFrame) -> Optional[int]:
        if cand.empty:
            return None
        if team_abbr:
            # try to filter by last name + team token in full_name (rarely available)
            # nba_api static data doesn't include team, so we mostly skip this step.
            pass
        act = cand[cand["is_active"] == True]
        row = (act if not act.empty else cand).iloc[0]
        return int(row["id"])

    # 1) exact
    exact = df[df["norm_name"] == norm]
    chosen = choose_one(exact)
    if chosen is not None:
        return chosen

    # 2) startswith
    starts = df[df["norm_name"].str.startswith(norm)]
    chosen = choose_one(starts)
    if chosen is not None:
        return chosen

    # 3) contains both first and last tokens
    parts = norm.split()
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
        contains = df[(df["norm_name"].str.contains(first)) & (df["norm_name"].str.contains(last))]
        chosen = choose_one(contains)
        if chosen is not None:
            return chosen

    # 4) first-name variant (e.g., nicolas → nic), then redo 1–3
    norm2 = _apply_firstname_variant(norm)
    if norm2 != norm:
        exact2 = df[df["norm_name"] == norm2]
        chosen = choose_one(exact2)
        if chosen is not None:
            return chosen

        starts2 = df[df["norm_name"].str.startswith(norm2)]
        chosen = choose_one(starts2)
        if chosen is not None:
            return chosen

        parts2 = norm2.split()
        if len(parts2) >= 2:
            first2, last2 = parts2[0], parts2[-1]
            contains2 = df[(df["norm_name"].str.contains(first2)) & (df["norm_name"].str.contains(last2))]
            chosen = choose_one(contains2)
            if chosen is not None:
                return chosen

    # 5) unique last-name fallback
    last = norm.split()[-1] if norm else ""
    if last:
        last_only = df[df["last"] == last]
        if len(last_only) == 1:
            return int(last_only.iloc[0]["id"])
        act_last = last_only[last_only["is_active"] == True]
        if len(act_last) == 1:
            return int(act_last.iloc[0]["id"])

    return None

# ----------------------------------------
# Game logs
# ----------------------------------------
def fetch_player_game_logs(
    player_id: int,
    season: str,
    season_type: str = "Regular Season",
    max_tries: int = 5,
    base_sleep: float = 0.7,
    cache_dir: str = "data/cache/gamelogs",
) -> pd.DataFrame:
    """
    Robust game-log fetch with retries and simple CSV cache.

    Cache path: data/cache/gamelogs/{player_id}_{season}_{season_type}.csv
    """
    import os, time, random

    os.makedirs(cache_dir, exist_ok=True)
    safe_season_type = season_type.replace(" ", "_")
    cache_path = os.path.join(cache_dir, f"{player_id}_{season}_{safe_season_type}.csv")

    # 1) serve from cache if present
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        # keep types/dates consistent
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        # coerce numerics we care about
        for c in ["PTS","REB","AST","FG3M","FG3A","FGA","FGM","STL","BLK","TOV","FTA"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # rebuild derived
        if "FGA" in df.columns and "FG3A" in df.columns:
            df["TWO_ATT"] = df["FGA"] - df["FG3A"]
        # ensure sort (most recent first)
        if "GAME_DATE" in df.columns:
            df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
        return df

    # 2) fetch with retries/backoff
    last_err = None
    for attempt in range(max_tries):
        try:
            gl = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star=season_type
            )
            df = gl.get_data_frames()[0]
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
            df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

            for c in ["PTS","REB","AST","FG3M","FG3A","FGA","FGM","STL","BLK","TOV","FTA"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            if "FGA" in df.columns and "FG3A" in df.columns:
                df["TWO_ATT"] = df["FGA"] - df["FG3A"]

            # save to cache
            df.to_csv(cache_path, index=False)
            return df
        except Exception as e:
            last_err = e
            # exponential backoff + jitter
            delay = base_sleep * (1.8 ** attempt) + random.uniform(0, 0.2)
            print(f"[retry] gamelog fetch failed for player {player_id} ({season} {season_type}) attempt {attempt+1}/{max_tries}: {e}. Sleeping {delay:.2f}s…")
            time.sleep(delay)

    # 3) if still failing, surface the last error
    raise last_err

def get_current_team_abbr(
    player_name: str,
    season: str,
    season_type: str = "Regular Season",
) -> str:
    """
    Resolve the player's current team from the most recent game log.
    Falls back to "" if not found.
    """
    pid = get_player_id_by_name(player_name)
    if pid is None:
        return ""
    try:
        df = fetch_player_game_logs(pid, season, season_type=season_type)
    except Exception:
        return ""
    if not df.empty and "TEAM_ABBREVIATION" in df.columns:
        val = str(df.iloc[0]["TEAM_ABBREVIATION"]).strip().upper()
        # basic sanity: NBA team abbrevs are typically 2–4 chars
        if 2 <= len(val) <= 4:
            return val
    return ""


# ----------------------------------------
# Prop → Series extraction
# ----------------------------------------
def _series_for_prop(df: pd.DataFrame, prop: str) -> pd.Series:
    """
    Return the series (most recent first) for a canonical prop.
    Prop must already be normalized upstream.
    """
    p = prop.upper().strip()

    # atomic direct columns
    direct_map = {
        "PTS": "PTS",
        "REB": "REB",
        "AST": "AST",
        "3PM": "FG3M",
        "FG3M": "FG3M",
        "3PA": "FG3A",
        "FG3A": "FG3A",
        "FGA": "FGA",
        "FGM": "FGM",
        "STL": "STL",
        "BLK": "BLK",
        "TOV": "TOV",
        "FTA": "FTA",
        "2PA": "TWO_ATT",   # derived earlier
    }
    if p in direct_map:
        col = direct_map[p]
        if col not in df.columns:
            raise RuntimeError(f"Column '{col}' not in gamelog for prop {p}.")
        return df[col]

    # combos
    if p == "PR":   # PTS+REB
        return df["PTS"] + df["REB"]
    if p == "PA":   # PTS+AST
        return df["PTS"] + df["AST"]
    if p == "RA":   # REB+AST
        return df["REB"] + df["AST"]
    if p == "PRA":  # PTS+REB+AST
        return df["PTS"] + df["REB"] + df["AST"]
    if p == "STOCKS":  # BLK + STL
        return df["BLK"] + df["STL"]

    raise ValueError(f"Unsupported canonical prop '{prop}'.")


def get_recent_stat_series(
    player_name: str,
    prop: str,
    season: str,
    look_back: int = 10,
    season_type: str = "Regular Season",
) -> pd.Series:
    """
    Return recent N values for a (possibly derived) prop.
    """
    player_id = get_player_id_by_name(player_name)
    if player_id is None:
        raise ValueError(f"Player '{player_name}' not found in nba_api directory.")

    logs = fetch_player_game_logs(player_id, season, season_type=season_type)
    s = _series_for_prop(logs, prop).astype(float).head(look_back)
    if s.empty:
        raise RuntimeError(f"No recent data for {player_name} {prop} in {season} ({season_type}).")
    return s


# ----------------------------------------
# Distribution estimation (Normal approx)
# ----------------------------------------
def estimate_normal_from_recent(
    player_name: str,
    prop: str,
    season: str,
    look_back: int = 10,
    season_type: str = "Regular Season",
    min_sigma: float = 1.25,
) -> Tuple[float, float, pd.Series]:
    """
    Estimate Normal(μ, σ) from the last N games of the canonical prop series.
    """
    s = get_recent_stat_series(
        player_name=player_name,
        prop=prop,
        season=season,
        look_back=look_back,
        season_type=season_type,
    )
    mu = float(s.mean())
    sigma = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    if sigma < min_sigma:
        sigma = min_sigma
    return mu, sigma, s
