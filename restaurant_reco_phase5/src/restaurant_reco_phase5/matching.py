from __future__ import annotations

from dataclasses import dataclass
from typing import List

from restaurant_reco_phase5.text_normalize import norm


@dataclass(frozen=True)
class LocationMatch:
    kind: str  # "exact" | "contains" | "none"
    score: float


def location_match(place: str, restaurant_location: str) -> LocationMatch:
    """
    Simple deterministic match against `location`.
    - exact (case/whitespace-insensitive) -> 1.0
    - contains either way -> 0.7
    - none -> 0.0
    """
    p = norm(place)
    r = norm(restaurant_location)
    if not p or not r:
        return LocationMatch(kind="none", score=0.0)
    if p == r:
        return LocationMatch(kind="exact", score=1.0)
    if p in r or r in p:
        return LocationMatch(kind="contains", score=0.7)
    return LocationMatch(kind="none", score=0.0)


def cuisine_overlap(preferred: List[str], restaurant_cuisines: List[str]) -> float:
    """
    Return overlap ratio in [0,1] relative to preferred cuisines.
    If preferred is empty -> 0 (no preference).
    """
    CUISINE_EXPANSION = {
        "south indian": ["south indian", "udupi", "andhra", "chettinad", "kerala"],
        "north indian": ["north indian", "punjabi", "mughlai", "awadhi"],
        "chinese": ["chinese", "asian", "thai", "momos"],
        "desserts": ["desserts", "ice cream", "bakery", "sweet shop", "beverages"],
    }
    
    expanded_preferred = set()
    for p in preferred:
        n_p = norm(p)
        if not n_p: continue
        
        # Check if the normalized preference matches an expansion key
        if n_p in CUISINE_EXPANSION:
            expanded_preferred.update(CUISINE_EXPANSION[n_p])
        else:
            expanded_preferred.add(n_p)

    rest = {norm(c) for c in restaurant_cuisines if norm(c)}
    if not expanded_preferred:
        return 0.0
        
    return len(expanded_preferred & rest) / float(len(expanded_preferred))

