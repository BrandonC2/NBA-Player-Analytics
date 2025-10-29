"""
CLI to generate optimal PrizePicks entries from board.csv.

It loads board.csv (from build_board_all.py), then finds the best combos for:
  - power2
  - flex3
  - flex5
  - flex6

This version ENFORCES that each leg's side and per-leg probability come from the board:
  - leg{i}_side   := board.chosen_side
  - leg{i}_p_hit  := board.p_hit_for_ev

Usage examples:
  python -m src.build_entries --board board.csv --entry_type power2 --min_p 0.55 --max_candidates 40 --top_n 50 --out entries_power2.csv
  python -m src.build_entries --board board.csv --entry_type flex5  --min_p 0.58 --max_candidates 36 --top_n 100 --out entries_flex5.csv
"""

import argparse
import re
import pandas as pd

from .optimizer import build_best_entries


def _standardize_key(s: str) -> str:
    """Lowercase + single spaces for robust matching."""
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _build_board_lookup(board: pd.DataFrame):
    """
    Create a fast lookup from (player, prop, line) -> row with chosen_side, p_hit_for_ev.
    We also normalize strings so minor spacing/casing differences don't break the match.
    """
    need_cols = {"player", "prop", "line", "chosen_side", "p_hit_for_ev"}
    missing = [c for c in need_cols if c not in board.columns]
    if missing:
        raise ValueError(f"board.csv missing required columns: {missing}")

    # Normalize copies for keying
    df = board.copy()
    df["__player_key"] = df["player"].map(_standardize_key)
    df["__prop_key"] = df["prop"].map(lambda x: str(x).strip().upper())
    # Use string form of line to be robust to 14.5 vs "14.5"
    df["__line_key"] = df["line"].astype(float).round(3)

    # Keep the most recent source_date if duplicates exist
    if "source_date" in df.columns:
        df["__source_ord"] = pd.to_datetime(df["source_date"], errors="coerce")
        df.sort_values("__source_ord", ascending=False, inplace=True)

    # Build dict
    lookup = {}
    for _, r in df.iterrows():
        key = (r["__player_key"], r["__prop_key"], r["__line_key"])
        if key not in lookup:
            lookup[key] = {
                "chosen_side": str(r["chosen_side"]).strip().upper(),
                "p_hit_for_ev": float(r["p_hit_for_ev"]),
            }
    return lookup


def _enforce_board_sides_and_probs(combos: pd.DataFrame, board_lookup: dict) -> pd.DataFrame:
    """
    For each leg{i}_*, overwrite side + p_hit from board lookup.
    If a leg can't be matched, we leave the existing values as-is but note a warning.
    """
    df = combos.copy()
    # detect legs by column name pattern
    leg_indices = set()
    for c in df.columns:
        m = re.match(r"leg(\d+)_player$", c)
        if m:
            leg_indices.add(int(m.group(1)))
    if not leg_indices:
        # Nothing to enforce; return as-is
        return df

    missing_matches = 0

    for i in sorted(leg_indices):
        pcol = f"leg{i}_player"
        propcol = f"leg{i}_prop"
        linecol = f"leg{i}_line"
        sidecol = f"leg{i}_side"
        phitcol = f"leg{i}_p_hit"

        # ensure columns exist
        for col in (sidecol, phitcol):
            if col not in df.columns:
                df[col] = ""

        # build keys
        pkey = df[pcol].map(_standardize_key)
        propkey = df[propcol].map(lambda x: str(x).strip().upper())
        linekey = df[linecol].astype(float).round(3)

        keys = list(zip(pkey, propkey, linekey))

        chosen_side_vals = []
        p_hit_vals = []

        for key, old_side, old_phit in zip(
            keys, df.get(sidecol, [""] * len(df)), df.get(phitcol, [None] * len(df))
        ):
            info = board_lookup.get(key)
            if info is None:
                # no match: keep existing values (optimizer result)
                chosen_side_vals.append(old_side)
                try:
                    p_hit_vals.append(float(old_phit))
                except Exception:
                    p_hit_vals.append(old_phit)
                missing_matches += 1
            else:
                chosen_side_vals.append(info["chosen_side"])
                p_hit_vals.append(float(info["p_hit_for_ev"]))

        df[sidecol] = chosen_side_vals
        df[phitcol] = p_hit_vals

    if missing_matches:
        print(f"[warn] build_entries: {missing_matches} leg(s) did not match board rows; kept optimizer values for those.")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build optimal PrizePicks entries from board.csv")
    parser.add_argument("--board", type=str, default="board.csv")
    parser.add_argument("--entry_type", type=str, required=True, help="power2|flex3|flex5|flex6")
    parser.add_argument("--min_p", type=float, default=0.55, help="minimum per-leg hit probability")
    parser.add_argument("--max_candidates", type=int, default=30, help="cap candidate legs before combinations")
    parser.add_argument("--top_n", type=int, default=50, help="number of top combos to output")
    parser.add_argument("--out", type=str, default="entries.csv")
    args = parser.parse_args()

    board = pd.read_csv(args.board)

    # Build combos via your optimizer
    combos = build_best_entries(
        board=board,
        entry_type=args.entry_type,
        min_p=args.min_p,
        max_candidates=args.max_candidates,
        top_n=args.top_n,
    )

    if combos.empty:
        print("[info] No combos built. Try lowering --min_p or increasing --max_candidates.")
        return

    # Enforce that each leg uses the board's chosen_side and p_hit_for_ev
    lookup = _build_board_lookup(board)
    combos = _enforce_board_sides_and_probs(combos, lookup)

    combos.to_csv(args.out, index=False)
    print(f"[ok] Wrote {len(combos)} {args.entry_type} entries → {args.out}")
    print("\nTop 10 preview:")
    print(combos.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
