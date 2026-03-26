from pathlib import Path

import pandas as pd
import pytest

from restaurant_reco_phase5.data_access import load_restaurants_parquet
from restaurant_reco_phase5.models import Budget, RecommendationRequest
from restaurant_reco_phase5.recommender import RecommenderError, recommend


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "restaurant_id": "r1",
                "name": "Alpha",
                "address": "Addr1",
                "location": "Banashankari",
                "cuisines": ["North Indian", "Chinese"],
                "rating": 4.2,
                "votes": 500,
                "cost_for_two": 800,
                "online_order": True,
                "book_table": False,
                "url": "http://x/1",
            },
            {
                "restaurant_id": "r2",
                "name": "Beta",
                "address": "Addr2",
                "location": "Banashankari",
                "cuisines": ["Italian"],
                "rating": 4.6,
                "votes": 50,
                "cost_for_two": 1200,
                "online_order": True,
                "book_table": True,
                "url": "http://x/2",
            },
            {
                "restaurant_id": "r3",
                "name": "Gamma",
                "address": "Addr3",
                "location": "Koramangala",
                "cuisines": ["North Indian"],
                "rating": 4.9,
                "votes": 1000,
                "cost_for_two": 700,
                "online_order": True,
                "book_table": True,
                "url": "http://x/3",
            },
        ]
    )


def test_recommend_basic_filters_and_ranking():
    df = _sample_df()
    req = RecommendationRequest(
        place="Banashankari",
        cuisines=["North Indian"],
        min_rating=4.0,
        budget=Budget(max=1000),
        top_n=5,
        debug=True,
    )
    resp = recommend(restaurants=df, request=req)
    assert resp.debug is not None
    assert resp.debug.candidate_count_before_filters == 3
    # only Alpha matches cuisine + location + budget + min_rating
    assert len(resp.recommendations) == 1
    assert resp.recommendations[0].name == "Alpha"
    assert resp.recommendations[0].score_breakdown is not None


def test_recommend_no_results_returns_empty():
    df = _sample_df()
    req = RecommendationRequest(place="Banashankari", cuisines=["Mexican"], top_n=5, debug=True)
    resp = recommend(restaurants=df, request=req)
    assert resp.recommendations == []
    assert resp.debug is not None
    assert resp.debug.candidate_count_after_filters == 0


def test_recommender_empty_dataset_error():
    with pytest.raises(RecommenderError):
        recommend(restaurants=pd.DataFrame(), request=RecommendationRequest(place="X"))


def test_load_restaurants_parquet_validation(tmp_path: Path):
    df = _sample_df()
    p = tmp_path / "restaurants.parquet"
    df.to_parquet(p, index=False)
    loaded = load_restaurants_parquet(p)
    assert len(loaded) == 3

