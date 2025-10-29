"""
Expected value (EV) helpers for PrizePicks-style entries.

We compute EV as the expected gross return multiple on a 1-unit stake,
assuming independent legs. Payout tables are configurable.

Notation:
- EV = sum_k P(exactly k hits) * payout_multiple_for_k
- For Power2: payout 3.0x if 2/2 hit, else 0
- For Flex5: {5:10.0, 4:2.0, 3:0.4}, etc.

We provide:
  - ev_from_probs(probs, payout_map)
  - ev_power2(p1, p2)
  - ev_flex_n_identical(p, n, payout_map)  # useful to score a single leg at different entry sizes
"""

from typing import Dict, Iterable, List
from math import comb


# ------------------------------
# Default payout tables (gross)
# Adjust here if PrizePicks changes payouts.
# ------------------------------
DEFAULT_PAYOUTS = {
    "power2": {2: 3.0},                           # both correct -> 3x return
    "flex3":  {3: 2.25, 2: 1.25},                  # 3-hit: 2.25x, 2-hit: 1.25x
    "flex5":  {5: 10.0, 4: 2.0, 3: 0.4},           # else 0
    "flex6":  {6: 25.0, 5: 2.0, 4: 0.4},           # else 0
}


# ------------------------------
# Core EV computation for arbitrary leg probabilities
# ------------------------------
def ev_from_probs(probs: Iterable[float], payout_map: Dict[int, float]) -> float:
    """
    Compute EV (gross return multiple) for an entry with independent legs.
    probs: iterable of leg hit probabilities [p1, p2, ...]
    payout_map: dict {k_hits: payout_multiple}

    Uses dynamic programming to compute distribution of total hits.
    """
    probs = list(probs)
    # pmf[k] = probability of exactly k hits so far
    pmf = [1.0]  # start with 0 legs, P(0 hits)=1
    for p in probs:
        next_pmf = [0.0] * (len(pmf) + 1)
        for k, pk in enumerate(pmf):
            # miss this leg
            next_pmf[k] += pk * (1.0 - p)
            # hit this leg
            next_pmf[k + 1] += pk * p
        pmf = next_pmf

    ev = 0.0
    for k, prob_k in enumerate(pmf):
        ev += prob_k * payout_map.get(k, 0.0)
    return ev


# ------------------------------
# Convenience wrappers
# ------------------------------
def ev_power2(p1: float, p2: float, payouts: Dict[str, Dict[int, float]] = DEFAULT_PAYOUTS) -> float:
    """EV for a 2-pick Power play (both must hit)."""
    return ev_from_probs([p1, p2], payouts["power2"])


def ev_flex_n_identical(p: float, n: int, payout_map: Dict[int, float]) -> float:
    """
    EV when *all* n legs share the same probability p (binomial shortcut).
    Useful to score a single leg vs. Flex5/Flex6 when companions are of similar quality.
    """
    ev = 0.0
    for k, mult in payout_map.items():
        ev += binomial_pmf(n, k, p) * mult
    return ev


# ------------------------------
# Binomial helper
# ------------------------------
def binomial_pmf(n: int, k: int, p: float) -> float:
    return comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


# ------------------------------
# Convenience accessors for common flex sizes
# ------------------------------
def ev_flex3_identical(p: float, payouts: Dict[str, Dict[int, float]] = DEFAULT_PAYOUTS) -> float:
    return ev_flex_n_identical(p, 3, payouts["flex3"])


def ev_flex5_identical(p: float, payouts: Dict[str, Dict[int, float]] = DEFAULT_PAYOUTS) -> float:
    return ev_flex_n_identical(p, 5, payouts["flex5"])


def ev_flex6_identical(p: float, payouts: Dict[str, Dict[int, float]] = DEFAULT_PAYOUTS) -> float:
    return ev_flex_n_identical(p, 6, payouts["flex6"])
