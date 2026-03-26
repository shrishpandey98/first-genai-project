from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from restaurant_reco_phase2.matching import cuisine_overlap, location_match
from restaurant_reco_phase2.models import (
    DebugInfo,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)
from restaurant_reco_phase2.scoring import ScoringWeights, score_candidate


class RecommenderError(ValueError):
    pass


def _apply_filters(df: pd.DataFrame, req: RecommendationRequest) -> Tuple[pd.DataFrame, Dict]:
    """
    Deterministic, interpretable filtering.
    Some filters are "hard" and others are "soft" via scoring; Phase 2 keeps
    it simple and deterministic.
    """
    filters_applied: Dict = {"place": req.place}
    out = df

    # Location is always required; keep candidates that at least "contain" match
    lm = out["location"].map(lambda x: location_match(req.place, x))
    out = out[lm.map(lambda m: m.score > 0.0)]

    if req.min_rating is not None:
        filters_applied["min_rating"] = req.min_rating
        # If rating is missing, exclude it (conservative for Phase 2)
        out = out[out["rating"].notna() & (out["rating"] >= req.min_rating)]

    if req.budget is not None and (req.budget.min is not None or req.budget.max is not None):
        filters_applied["budget"] = req.budget.model_dump()
        # If cost missing, exclude it (conservative for Phase 2)
        if req.budget.min is not None:
            out = out[out["cost_for_two"].notna() & (out["cost_for_two"] >= req.budget.min)]
        if req.budget.max is not None:
            out = out[out["cost_for_two"].notna() & (out["cost_for_two"] <= req.budget.max)]

    if req.online_order is not None:
        filters_applied["online_order"] = req.online_order
        out = out[out["online_order"].notna() & (out["online_order"] == req.online_order)]

    if req.book_table is not None:
        filters_applied["book_table"] = req.book_table
        out = out[out["book_table"].notna() & (out["book_table"] == req.book_table)]

    if req.cuisines:
        # Hard cuisine filter: require at least one overlap (Phase 2 requirement)
        filters_applied["cuisines"] = list(req.cuisines)
        overlaps = out["cuisines"].map(lambda cs: cuisine_overlap(req.cuisines, cs) > 0.0)
        out = out[overlaps]

    return out, filters_applied


def recommend(
    *,
    restaurants: pd.DataFrame,
    request: RecommendationRequest,
    weights: ScoringWeights = ScoringWeights(),
) -> RecommendationResponse:
    if restaurants is None or len(restaurants) == 0:
        raise RecommenderError("restaurants dataset is empty")

    before = len(restaurants)
    filtered, filters_applied = _apply_filters(restaurants, request)
    after = len(filtered)

    if after == 0:
        # Deterministic empty response with debug info.
        dbg = DebugInfo(
            filters_applied=filters_applied,
            candidate_count_before_filters=before,
            candidate_count_after_filters=after,
        )
        return RecommendationResponse(recommendations=[], debug=dbg if request.debug else None)

    scored_rows = []
    for _, row in filtered.iterrows():
        score, breakdown, loc = score_candidate(
            place=request.place,
            preferred_cuisines=request.cuisines,
            budget=request.budget,
            rating=(None if pd.isna(row["rating"]) else float(row["rating"])),
            votes=(None if pd.isna(row["votes"]) else int(row["votes"])),
            cost_for_two=(None if pd.isna(row["cost_for_two"]) else int(row["cost_for_two"])),
            restaurant_location=str(row["location"]),
            restaurant_cuisines=list(row["cuisines"] or []),
            weights=weights,
        )

        reasons: List[str] = []
        if loc.kind != "none":
            reasons.append(f"Location match: {loc.kind.replace('_', ' ')}")
        if request.min_rating is not None and not pd.isna(row["rating"]):
            reasons.append(f"Meets min rating: {row['rating']}")
        if request.budget is not None and not pd.isna(row["cost_for_two"]):
            reasons.append(f"Within budget: {row['cost_for_two']}")
        if request.cuisines:
            reasons.append("Cuisine match")
        if request.online_order is True:
            reasons.append("Supports online order")
        if request.book_table is True:
            reasons.append("Supports table booking")

        item = RecommendationItem(
            restaurant_id=str(row["restaurant_id"]),
            name=str(row["name"]),
            address=str(row["address"]),
            location=str(row["location"]),
            cuisines=list(row["cuisines"] or []),
            rating=(None if pd.isna(row["rating"]) else float(row["rating"])),
            votes=(None if pd.isna(row["votes"]) else int(row["votes"])),
            cost_for_two=(None if pd.isna(row["cost_for_two"]) else int(row["cost_for_two"])),
            online_order=(None if pd.isna(row["online_order"]) else bool(row["online_order"])),
            book_table=(None if pd.isna(row["book_table"]) else bool(row["book_table"])),
            url=str(row["url"]),
            reasons=reasons,
            score=float(score),
            score_breakdown=breakdown if request.debug else None,
        )
        scored_rows.append(item)

    scored_rows.sort(key=lambda x: x.score, reverse=True)
    top = scored_rows[: request.top_n]

    dbg = DebugInfo(
        filters_applied=filters_applied,
        candidate_count_before_filters=before,
        candidate_count_after_filters=after,
    )

    return RecommendationResponse(recommendations=top, debug=dbg if request.debug else None)

