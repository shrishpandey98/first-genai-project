import pytest

from restaurant_reco_phase4.models import Budget, RecommendationRequest


def test_budget_range_validation():
    with pytest.raises(ValueError):
        Budget(min=1000, max=500)


def test_request_cuisine_dedup_and_strip():
    req = RecommendationRequest(place=" Banashankari ", cuisines=["North Indian", " north indian ", ""])
    assert req.place == "Banashankari"
    assert req.cuisines == ["North Indian"]


def test_request_top_n_bounds():
    with pytest.raises(ValueError):
        RecommendationRequest(place="X", top_n=0)

