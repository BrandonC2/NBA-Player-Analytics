"""
Build a daily board with:
- player, team, opponent (from schedule)
- prop normalization (maps PP/CSV labels to canonical props)
- line
- recent mu/sigma (Normal approx from logs)
- P(Over), P(Under)
- EV-like scores for Flex3/Flex5/Flex6 (identical-prob approximation for ranking)

Typical:
  python -m src.build_board_all \
    --date 2025-10-27 \
    --season 2025-26 \
    --season_type "Regular Season" \
    --look_back 10 \
    --in_csv data/prizepicks_lines.csv \
    --use_prizepicks 1 \
    --merge_mode replace \
    --out board.csv \
    --debug 1
"""

import argparse
import re
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

from .fetch_data import fetch_games_by_date
from .compute_metrics import estimate_normal_from_recent, get_current_team_abbr
from .ev_calc import ev_flex3_identical, ev_flex5_identical, ev_flex6_identical
from .prizepicks_connector import fetch_prizepicks_projections


# ------------------------------
# Prop normalization
# ------------------------------

_CANON_ALIASES = {
    # canonical
    "PTS": "PTS", "POINTS": "PTS",
    "REB": "REB", "REBOUNDS": "REB",
    "AST": "AST", "ASSISTS": "AST",

    # 3-pointers
    "3PM": "3PM", "FG3M": "3PM",
    "3PT": "3PM", "3PT MADE": "3PM", "3-PT MADE": "3PM",
    "THREE POINTERS MADE": "3PM", "THREE POINTS MADE": "3PM",
    "3-PT FIELD GOALS MADE": "3PM",

    "3PA": "3PA", "FG3A": "3PA",
    "3PT ATTEMPTED": "3PA", "3-PT ATTEMPTED": "3PA",
    "THREE POINTERS ATTEMPTED": "3PA", "THREE POINTS ATTEMPTED": "3PA",

    # field goals
    "FGA": "FGA", "FIELD GOALS ATTEMPTED": "FGA", "FG ATTEMPTED": "FGA",
    "FGM": "FGM", "FIELD GOALS MADE": "FGM", "FG MADE": "FGM",

    # misc singles
    "STL": "STL", "STEALS": "STL",
    "BLK": "BLK", "BLOCKS": "BLK", "BLOCKED SHOTS": "BLK",
    "TOV": "TOV", "TURNOVERS": "TOV",
    "FTA": "FTA", "FREE THROWS ATTEMPTED": "FTA",
    "2PA": "2PA", "TWO POINTERS ATTEMPTED": "2PA",

    # combos
    "PR": "PR", "PTS+REBS": "PR",
    "PA": "PA", "PTS+ASTS": "PA",
    "RA": "RA", "REBS+ASTS": "RA",
    "PRA": "PRA", "PTS+REBS+ASTS": "PRA",
    "STOCKS": "STOCKS", "BLKS+STLS": "STOCKS",

    # unsupported (explicitly map to None so diagnostics show why they drop)
    "FANTASY SCORE": None, "FANTASY": None, "DFS": None,
}

def normalize_prop(raw: str) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().upper()
    s = re.sub(r"\s+", " ", s)  # collapse whitespace
    return _CANON_ALIASES.get(s, s if s in _CANON_ALIASES else None)


# ------------------------------
# matchup helpers
# ------------------------------
def build_opponent_map(games_df: pd.DataFrame) -> dict:
    opp = {}
    for _, row in games_df.iterrows():
        team = str(row["TEAM_ABBREVIATION"]).strip()
        matchup = str(row["MATCHUP"])
        other = parse_opponent_from_matchup(matchup, team)
        if team and other:
            opp[team] = other
    return opp

def parse_opponent_from_matchup(matchup: str, team_abbr: str) -> Optional[str]:
    m = matchup.replace(".", "")
    tokens = re.split(r"\s+vs\s+|\s+@\s+|\s+VS\s+|\s+Vs\s+", m)
    for t in tokens:
        t = t.strip()
        if t and t != team_abbr:
            return t
    return None


# ------------------------------
# probability helpers (normal approx)
# ------------------------------
def normal_cdf(x: float, mu: float, sigma: float) -> float:
    from math import erf, sqrt
    z = (x - mu) / sigma
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))

def p_over_under(line: float, mu: float, sigma: float) -> Tuple[float, float]:
    p_under = normal_cdf(line, mu, sigma)
    p_over = 1.0 - p_under
    return p_over, p_under


# ------------------------------
# dedupe: pick best alt line per (player, prop)
# ------------------------------
def dedupe_board(board: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        return board

    def pick_one(group: pd.DataFrame) -> pd.Series:
        g = group.copy()
        for col in ["ev_flex6_identical", "ev_flex5_identical", "p_hit_for_ev", "line"]:
            if col not in g.columns:
                g[col] = 0.0
        g = g.sort_values(
            by=["ev_flex6_identical", "ev_flex5_identical", "p_hit_for_ev", "line"],
            ascending=[False, False, False, True],
        )
        return g.iloc[0]

    return (
        board.groupby(["player", "prop"], as_index=False, group_keys=False)
             .apply(pick_one, include_groups=False)  # silence pandas 2.x FutureWarning
             .reset_index(drop=True)
    )


# ------------------------------
# main build
# ------------------------------
def build_board(
    date: str,
    season: str,
    in_csv: str,
    out_csv: str,
    look_back: int = 10,
    season_type: str = "Regular Season",
    min_sigma: float = 1.25,
    use_prizepicks: bool = False,
    merge_mode: str = "replace",   # "replace" | "union"
    debug: bool = True,
) -> pd.DataFrame:
    # 1) Opponent mapping for the date
    try:
        games_df = fetch_games_by_date(date, season)
    except Exception as e:
        print("[warn] fetch_games_by_date failed:", e)
        games_df = pd.DataFrame(columns=["TEAM_ABBREVIATION", "MATCHUP"])
    opp_map = build_opponent_map(games_df)

    # 2) Input lines
    manual = pd.read_csv(in_csv) if in_csv else pd.DataFrame(columns=["player","team","prop","line"])
    manual.columns = [c.lower() for c in manual.columns]

    live = pd.DataFrame()
    if use_prizepicks:
        try:
            live = fetch_prizepicks_projections("NBA", league_id=None, per_page=800)
        except Exception as e:
            print("[warn] PrizePicks live fetch failed:", e)
            live = pd.DataFrame(columns=["player","team","prop","line","source_date"])

    keep_cols = ["player","team","prop","line","source_date","direction","side_lock","pp_tags"]
    for df in (manual, live):
        for c in keep_cols:
            if c not in df.columns:
                df[c] = "" if c not in ("line",) else None

    if debug:
        print(f"[dbg] manual rows: {len(manual)} | live rows: {len(live)}")

    # Merge policy with safe fallback
    if use_prizepicks and merge_mode == "replace" and live.empty and not manual.empty:
        if debug:
            print("[dbg] live is empty; falling back to manual lines (replace→manual).")
        lines = manual.copy()
    elif use_prizepicks:
        if merge_mode == "replace":
            lines = live.copy()
        else:
            lines = pd.concat([live, manual], ignore_index=True)
            lines = lines.drop_duplicates(subset=["player","team","prop","line"], keep="first")
    else:
        lines = manual

    if lines.empty:
        if debug:
            print("[dbg] lines is empty before normalization. Returning empty board.")
        pd.DataFrame().to_csv(out_csv, index=False)
        return pd.DataFrame()

    # Normalize props and drop unsupported
    if debug:
        try:
            print("[dbg] sample props (pre-normalize):",
                  lines["prop"].astype(str).str.upper().value_counts().head(12).to_dict())
        except Exception:
            pass

    lines["prop_norm"] = lines["prop"].apply(normalize_prop)
    before = len(lines)
    lines = lines[~lines["prop_norm"].isna()].copy()
    if debug:
        print(f"[dbg] after normalize: kept {len(lines)} rows, dropped {before - len(lines)}.")
        if len(lines):
            print("[dbg] sample normalized props:", lines["prop_norm"].value_counts().head(12).to_dict())

    # 3) Compute metrics per line
    rows = []
    skipped_metrics = 0
    skipped_lineparse = 0

    for _, r in lines.iterrows():
        player = str(r["player"]).strip()
        team = str(r.get("team", "")).strip().upper()
        prop = str(r["prop_norm"]).strip().upper()

        # parse line + direction
        try:
            line_val = float(r["line"])
        except Exception:
            skipped_lineparse += 1
            continue

        direction = str(r.get("direction", "")).strip().upper()
        side_lock = str(r.get("side_lock", "")).strip().upper()  # Demon/Goblin → "OVER"
        pp_tags = str(r.get("pp_tags", "")).strip()

        # μ/σ from recent logs for canonical prop
        try:
            mu, sigma, _ = estimate_normal_from_recent(
                player_name=player,
                prop=prop,
                season=season,
                look_back=look_back,
                season_type=season_type,
                min_sigma=min_sigma,
            )
        except Exception as e:
            if debug:
                print(f"[warn] metrics failed for {player} {r.get('prop','')}: {e}")
            skipped_metrics += 1
            continue

        # resolve team from latest NBA log (overrides PP when available)
        resolved_team = get_current_team_abbr(player, season=season, season_type=season_type)
        if resolved_team:
            team = resolved_team

        # probabilities
        p_over, p_under = p_over_under(line_val, mu, sigma)

        # precedence: tag lock -> manual direction -> model best
        if side_lock in ("OVER", "UNDER"):
            chosen = side_lock
        elif direction in ("OVER", "UNDER"):
            chosen = direction
        else:
            chosen = "OVER" if p_over >= p_under else "UNDER"

        p_hit = p_over if chosen == "OVER" else p_under

        # EV approximations
        ev3 = ev_flex3_identical(p_hit)
        ev5 = ev_flex5_identical(p_hit)
        ev6 = ev_flex6_identical(p_hit)

        # opponent for the date (if the team plays today)
        opponent = opp_map.get(team, "")

        rows.append({
            "date": date,
            "player": player,
            "team": team,
            "opponent": opponent,
            "prop": prop,                 # canonical
            "display_prop": r["prop"],    # original text from PP/CSV
            "line": line_val,
            "mu": round(mu, 3),
            "sigma": round(sigma, 3),
            "p_over": round(p_over, 6),
            "p_under": round(p_under, 6),
            "chosen_side": chosen,
            "p_hit_for_ev": round(p_hit, 6),
            "ev_flex3_identical": round(ev3, 6),
            "ev_flex5_identical": round(ev5, 6),
            "ev_flex6_identical": round(ev6, 6),
            "source_date": r.get("source_date", ""),
            "pp_tags": pp_tags,
            "side_lock": side_lock,
        })

    if debug:
        print(f"[dbg] built {len(rows)} rows | skipped_lineparse={skipped_lineparse} | skipped_metrics={skipped_metrics}")

    board = pd.DataFrame(rows)

    if not board.empty:
        board = board.sort_values(
            by=["ev_flex6_identical", "ev_flex5_identical", "p_hit_for_ev"],
            ascending=False
        ).reset_index(drop=True)
        board = dedupe_board(board)

    board.to_csv(out_csv, index=False)
    return board


def main():
    parser = argparse.ArgumentParser(description="Build daily board with probabilities and EV-like scores")
    parser.add_argument("--date", type=str, default=datetime.utcnow().date().isoformat())
    parser.add_argument("--season", type=str, default="2025-26")
    parser.add_argument("--season_type", type=str, default="Regular Season")
    parser.add_argument("--look_back", type=int, default=10)
    parser.add_argument("--in_csv", type=str, default="data/prizepicks_lines.csv")
    parser.add_argument("--out", type=str, default="board.csv")
    parser.add_argument("--use_prizepicks", type=int, default=1)
    parser.add_argument("--merge_mode", type=str, default="replace", choices=["replace","union"])
    parser.add_argument("--debug", type=int, default=1)
    args = parser.parse_args()

    board = build_board(
        date=args.date,
        season=args.season,
        in_csv=args.in_csv,
        out_csv=args.out,
        look_back=args.look_back,
        season_type=args.season_type,
        use_prizepicks=bool(args.use_prizepicks),
        merge_mode=args.merge_mode,
        debug=bool(args.debug),
    )

    print("\n=== Built Board ===")
    print(board.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
