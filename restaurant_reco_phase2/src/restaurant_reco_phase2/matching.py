from __future__ import annotations

from dataclasses import dataclass
from typing import List

from restaurant_reco_phase2.text_normalize import norm


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
    if not preferred:
        return 0.0
    pref = {norm(c) for c in preferred if norm(c)}
    rest = {norm(c) for c in restaurant_cuisines if norm(c)}
    if not pref:
        return 0.0
    return len(pref & rest) / float(len(pref))

