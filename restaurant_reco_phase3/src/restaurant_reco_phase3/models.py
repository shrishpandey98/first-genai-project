from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Budget(BaseModel):
    """
    Budget constraint for cost_for_two (in INR).

    - If `max` is set, candidates with cost_for_two <= max are preferred/allowed.
    - If `min` is set, candidates with cost_for_two >= min are preferred/allowed.
    - If both are set, min <= cost_for_two <= max.
    """

    min: Optional[int] = Field(default=None, ge=0)
    max: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_range(self) -> "Budget":
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("budget.min cannot be greater than budget.max")
        return self


class RecommendationRequest(BaseModel):
    place: Annotated[str, Field(min_length=1, description="Neighborhood/area to search in.")]
    cuisines: List[str] = Field(default_factory=list, description="Preferred cuisines.")
    min_rating: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    budget: Optional[Budget] = None
    online_order: Optional[bool] = None
    book_table: Optional[bool] = None
    top_n: int = Field(default=5, ge=1, le=50)
    debug: bool = False

    @field_validator("place")
    @classmethod
    def _strip_place(cls, v: str) -> str:
        return v.strip()

    @field_validator("cuisines")
    @classmethod
    def _normalize_cuisines(cls, v: List[str]) -> List[str]:
        out: list[str] = []
        for c in v:
            c2 = str(c).strip()
            if c2:
                out.append(c2)
        # preserve order, remove duplicates
        seen: set[str] = set()
        deduped: list[str] = []
        for c in out:
            key = c.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        return deduped


class ScoreBreakdown(BaseModel):
    total: float
    rating: float
    votes: float
    budget_fit: float
    cuisine_match: float
    location_match: float


class RecommendationItem(BaseModel):
    restaurant_id: str
    name: str
    address: str
    location: str
    cuisines: List[str]
    rating: Optional[float]
    votes: Optional[int]
    cost_for_two: Optional[int]
    online_order: Optional[bool]
    book_table: Optional[bool]
    url: str
    reasons: List[str]
    score: float
    score_breakdown: Optional[ScoreBreakdown] = None
    explanation: Optional[str] = None


class DebugInfo(BaseModel):
    filters_applied: Dict[str, Any]
    candidate_count_before_filters: int
    candidate_count_after_filters: int
    scoring_version: Literal["v1"] = "v1"


class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]
    debug: Optional[DebugInfo] = None
    summary: Optional[str] = None

