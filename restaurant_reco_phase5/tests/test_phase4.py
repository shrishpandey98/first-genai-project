import pandas as pd

from restaurant_reco_phase5.matching import cuisine_overlap
from restaurant_reco_phase5.models import RecommendationRequest
from restaurant_reco_phase5.recommender import recommend


def test_cuisine_expansion():
    # Because 'south indian' expands to ['south indian', 'udupi', ...],
    # if the restaurant has 'udupi', it should match.
    score = cuisine_overlap(["south indian"], ["udupi", "chinese"])
    assert score > 0.0, "Udupi should match South Indian due to expansion"


def test_recommender_semantic_ordering():
    df = pd.DataFrame([
        {
            "restaurant_id": "r1", "name": "Bland Place", "address": "123 St",
            "location": "Banashankari", "cuisines": ["North Indian"],
            "rating": 4.0, "votes": 100, "cost_for_two": 500,
            "online_order": True, "book_table": True, "url": "http://1"
        },
        {
            "restaurant_id": "r2", "name": "Cozy Romantic Italian", "address": "123 St",
            "location": "Banashankari", "cuisines": ["Italian"],
            "rating": 4.0, "votes": 100, "cost_for_two": 500,
            "online_order": True, "book_table": True, "url": "http://2"
        }
    ])

    req = RecommendationRequest(
        place="Banashankari",
        # Notice we don't specify cuisine here, so it relies on semantic match
        free_text="Looking for a cozy place for a romantic date night",
        top_n=2,
        debug=True
    )
    
    resp = recommend(restaurants=df, request=req)
    
    recs = resp.recommendations
    assert len(recs) == 2
    
    # "Cozy Romantic Italian" should rank higher semantically
    # than "Bland Place" given the free text
    assert recs[0].name == "Cozy Romantic Italian"
    assert recs[0].score_breakdown.semantic_match > recs[1].score_breakdown.semantic_match
