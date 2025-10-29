"""
Probability calculator for Over/Under on a manual line
using a Normal approximation from recent game logs.

CLI:
  python -m src.prob_calc --player "Stephen Curry" --prop PTS --line 25.5 --season 2025-26 --look_back 10
"""

import argparse
from math import erf, sqrt

from .compute_metrics import estimate_normal_from_recent


# ----------------------------------------
# Normal CDF (no SciPy required)
# ----------------------------------------
def normal_cdf(x: float, mu: float, sigma: float) -> float:
    """Φ((x-μ)/σ)"""
    z = (x - mu) / sigma
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def prob_over_under(line: float, mu: float, sigma: float):
    """
    Return P(Over), P(Under) for a continuous threshold.
    We treat "Over" as strictly > line (approx with CDF at line).
    """
    p_under = normal_cdf(line, mu, sigma)
    p_over = 1.0 - p_under
    return p_over, p_under


def main():
    parser = argparse.ArgumentParser(description="Over/Under probability from recent logs (Normal approx)")
    parser.add_argument("--player", type=str, required=True, help="e.g., 'Stephen Curry'")
    parser.add_argument("--prop", type=str, required=True, help="PTS | REB | AST | 3PM")
    parser.add_argument("--line", type=float, required=True, help="e.g., 25.5")
    parser.add_argument("--season", type=str, default="2025-26", help="e.g., 2025-26")
    parser.add_argument("--season_type", type=str, default="Regular Season", help="Regular Season | Pre Season | Playoffs")
    parser.add_argument("--look_back", type=int, default=10, help="number of most recent games to use")
    parser.add_argument("--min_sigma", type=float, default=1.25, help="variance floor (std) to avoid zero-variance")
    args = parser.parse_args()

    mu, sigma, series_used = estimate_normal_from_recent(
        player_name=args.player,
        prop=args.prop,
        season=args.season,
        look_back=args.look_back,
        season_type=args.season_type,
        min_sigma=args.min_sigma,
    )
    p_over, p_under = prob_over_under(args.line, mu, sigma)

    # Pretty print
    print(f"\nPlayer       : {args.player}")
    print(f"Prop         : {args.prop}")
    print(f"Line         : {args.line}")
    print(f"Season       : {args.season} ({args.season_type})")
    print(f"Look-back    : last {args.look_back} games")
    print(f"μ (mean)     : {mu:.3f}")
    print(f"σ (std floor): {sigma:.3f}")
    print(f"P(Over)      : {p_over:.3%}")
    print(f"P(Under)     : {p_under:.3%}")
    print("\nRecent values (most recent first):")
    print(series_used.to_string(index=False))


if __name__ == "__main__":
    main()
