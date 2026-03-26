from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from restaurant_reco_phase5.matching import LocationMatch, cuisine_overlap, location_match
from restaurant_reco_phase5.models import Budget, ScoreBreakdown


@dataclass(frozen=True)
class ScoringWeights:
    rating: float = 0.35
    votes: float = 0.10
    budget_fit: float = 0.15
    cuisine_match: float = 0.20
    location_match: float = 0.10
    semantic_match: float = 0.10


def _normalize_rating(r: Optional[float]) -> float:
    if r is None:
        return 0.0
    # Map 0..5 to 0..1
    return max(0.0, min(1.0, r / 5.0))


def _normalize_votes(votes: Optional[int]) -> float:
    if votes is None or votes <= 0:
        return 0.0
    # log scale: 0..~1 as votes grow
    return max(0.0, min(1.0, math.log10(votes + 1) / 5.0))


def _budget_fit(cost_for_two: Optional[int], budget: Optional[Budget]) -> float:
    """
    Return a soft budget fit score in [0,1].
    - If no budget or missing cost -> 0 (no signal).
    - If within range -> 1.
    - If outside -> decays with relative distance.
    """
    if budget is None or cost_for_two is None:
        return 0.0

    bmin = budget.min
    bmax = budget.max
    if bmin is None and bmax is None:
        return 0.0

    if bmin is not None and bmax is not None and bmin <= cost_for_two <= bmax:
        return 1.0
    if bmax is not None and cost_for_two <= bmax:
        return 1.0
    if bmin is not None and cost_for_two >= bmin:
        return 1.0

    # Outside constraints: score decays with relative distance from nearest boundary
    if bmax is not None and cost_for_two > bmax:
        diff = cost_for_two - bmax
        return max(0.0, 1.0 - (diff / max(1.0, float(bmax))))
    if bmin is not None and cost_for_two < bmin:
        diff = bmin - cost_for_two
        return max(0.0, 1.0 - (diff / max(1.0, float(bmin))))
    return 0.0


def score_candidate(
    *,
    place: str,
    preferred_cuisines: List[str],
    budget: Optional[Budget],
    rating: Optional[float],
    votes: Optional[int],
    cost_for_two: Optional[int],
    restaurant_location: str,
    restaurant_cuisines: List[str],
    semantic_score: float = 0.0,
    weights: ScoringWeights = ScoringWeights(),
) -> Tuple[float, ScoreBreakdown, LocationMatch]:
    loc = location_match(place, restaurant_location)
    cuisine = cuisine_overlap(preferred_cuisines, restaurant_cuisines)
    r = _normalize_rating(rating)
    v = _normalize_votes(votes)
    b = _budget_fit(cost_for_two, budget)

    total = (
        weights.rating * r
        + weights.votes * v
        + weights.budget_fit * b
        + weights.cuisine_match * cuisine
        + weights.location_match * loc.score
        + weights.semantic_match * semantic_score
    )

    breakdown = ScoreBreakdown(
        total=float(total),
        rating=float(weights.rating * r),
        votes=float(weights.votes * v),
        budget_fit=float(weights.budget_fit * b),
        cuisine_match=float(weights.cuisine_match * cuisine),
        location_match=float(weights.location_match * loc.score),
        semantic_match=float(weights.semantic_match * semantic_score),
    )
    return float(total), breakdown, loc

