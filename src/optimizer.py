"""
Entry optimizer for PrizePicks-style combos.

Given a board.csv with columns:
  player, team, opponent, prop, line, p_over, p_under, chosen_side, p_hit_for_ev, ...
we will:
  - select top candidate legs by their per-leg probability (configurable)
  - build combinations for entry_type (flex3|flex5|flex6|power2)
  - compute EV using each leg's *own* hit probability (no identical-prob approx)
  - output a DataFrame of top combos sorted by EV (gross multiple)

Usage (called from build_entries.py):
  from optimizer import build_best_entries
"""

import itertools
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from .ev_calc import DEFAULT_PAYOUTS, ev_from_probs


# ------------------------------
# Helpers
# ------------------------------
def _payout_map_for(entry_type: str) -> Dict[int, float]:
    entry_type = entry_type.lower()
    if entry_type == "power2":
        return DEFAULT_PAYOUTS["power2"]
    if entry_type == "flex3":
        return DEFAULT_PAYOUTS["flex3"]
    if entry_type == "flex5":
        return DEFAULT_PAYOUTS["flex5"]
    if entry_type == "flex6":
        return DEFAULT_PAYOUTS["flex6"]
    raise ValueError(f"Unsupported entry_type '{entry_type}'. Use power2|flex3|flex5|flex6.")


def _k_for(entry_type: str) -> int:
    entry_type = entry_type.lower()
    return {
        "power2": 2,
        "flex3": 3,
        "flex5": 5,
        "flex6": 6,
    }[entry_type]


def _leg_probability(row: pd.Series) -> float:
    """
    Use the probability that corresponds to the row's chosen_side.
    If chosen_side missing, fall back to max(p_over, p_under).
    """
    side = str(row.get("chosen_side", "")).upper()
    p_over = float(row["p_over"])
    p_under = float(row["p_under"])
    if side == "OVER":
        return p_over
    if side == "UNDER":
        return p_under
    return max(p_over, p_under)


def _leg_side(row: pd.Series) -> str:
    side = str(row.get("chosen_side", "")).upper()
    if side in ("OVER", "UNDER"):
        return side
    return "OVER" if float(row["p_over"]) >= float(row["p_under"]) else "UNDER"


def _unique_leg_key(row: pd.Series) -> Tuple[str, str]:
    """
    Uniqueness guard to avoid duplicate player/prop legs in the same entry.
    """
    return (str(row["player"]).strip(), str(row["prop"]).strip().upper())


def _flatten_combo_rows(combo_rows: List[pd.Series], entry_type: str, ev: float) -> dict:
    """
    Produce a single dict row for the combo, flattening legs into fixed columns.
    """
    out = {
        "entry_type": entry_type,
        "k": len(combo_rows),
        "ev": round(ev, 6),
    }
    # include a quick combined name for easy reading
    out["legs"] = " | ".join(
        f"{r['player']} {r['prop']} { _leg_side(r) } {r['line']}"
        for r in combo_rows
    )
    # flatten up to 6 legs
    for i, r in enumerate(combo_rows, 1):
        out[f"leg{i}_player"] = r["player"]
        out[f"leg{i}_team"] = r.get("team", "")
        out[f"leg{i}_opponent"] = r.get("opponent", "")
        out[f"leg{i}_prop"] = r["prop"]
        out[f"leg{i}_line"] = r["line"]
        out[f"leg{i}_side"] = _leg_side(r)
        out[f"leg{i}_p_hit"] = round(_leg_probability(r), 6)
    return out


# ------------------------------
# Public API
# ------------------------------
def select_candidates(
    board: pd.DataFrame,
    min_p: float = 0.55,
    max_candidates: int = 30,
) -> pd.DataFrame:
    """
    Filter board rows by probability and keep top N by p_hit_for_ev.
    Ensures each (player, prop) appears once (keeps best side/line from board).
    """
    df = board.copy()
    if "p_hit_for_ev" not in df.columns:
        # derive quickly if missing
        df["p_hit_for_ev"] = df.apply(_leg_probability, axis=1)

    # one row per (player, prop) with max p_hit
    df["key"] = df.apply(lambda r: _unique_leg_key(r), axis=1)
    df = df.sort_values("p_hit_for_ev", ascending=False).drop_duplicates("key")

    # filter and cap
    df = df[df["p_hit_for_ev"] >= min_p].head(max_candidates).reset_index(drop=True)
    df = df.drop(columns=["key"], errors="ignore")
    return df


def build_best_entries(
    board: pd.DataFrame,
    entry_type: str,
    min_p: float = 0.55,
    max_candidates: int = 30,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Build best entries of a given type from a board of legs.

    entry_type: power2|flex3|flex5|flex6
    min_p:      minimum leg hit probability to consider
    max_candidates: cap on candidate legs to keep combinations tractable
    top_n:      number of top combos to return
    """
    entry_type = entry_type.lower()
    k = _k_for(entry_type)
    payout_map = _payout_map_for(entry_type)

    candidates = select_candidates(board, min_p=min_p, max_candidates=max_candidates)
    if candidates.empty or len(candidates) < k:
        return pd.DataFrame()

    # precompute p_hit for speed
    candidates = candidates.copy()
    candidates["p_hit"] = candidates.apply(_leg_probability, axis=1)

    rows = []
    # iterate combinations (watch combinatorial growth)
    for combo in itertools.combinations(candidates.itertuples(index=False), k):
        # enforce unique players (no two legs same player)
        uniq_players = {getattr(r, "player") for r in combo}
        if len(uniq_players) != k:
            continue

        probs = [getattr(r, "p_hit") for r in combo]
        ev = ev_from_probs(probs, payout_map)
        # assemble flattened row
        combo_df_rows = [pd.Series({col: getattr(r, col) for col in candidates.columns}) for r in combo]
        rows.append(_flatten_combo_rows(combo_df_rows, entry_type, ev))

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("ev", ascending=False).head(top_n).reset_index(drop=True)
    return out
