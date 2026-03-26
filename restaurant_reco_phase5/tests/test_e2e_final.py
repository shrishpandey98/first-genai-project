import os
import pandas as pd
from restaurant_reco_phase5.models import RecommendationRequest
from restaurant_reco_phase5.recommender import recommend

def test_full_system_e2e():
    # 1. Load the real processed data
    DATA_PATH = "/Users/shrishpandey/Desktop/cursor_project/first-genai-project/restaurant_reco/data/processed/restaurants.parquet"
    assert os.path.exists(DATA_PATH), f"Data not found at {DATA_PATH}"
    df = pd.read_parquet(DATA_PATH)
    
    # 2. Setup a complex request (Semantic + Filter)
    req = RecommendationRequest(
        place="Banashankari",
        free_text="I want a cozy and romantic place for a date night which is cheap and affordable",
        top_n=3,
        debug=True
    )
    
    # 3. Call recommend
    print("\n[E2E] Running recommendation engine...")
    resp = recommend(restaurants=df, request=req)
    
    # 4. Assertions
    assert len(resp.recommendations) > 0, "No recommendations returned"
    
    top_resto = resp.recommendations[0]
    print(f"[E2E] Top Recommendation: {top_resto.name}")
    print(f"[E2E] Score: {top_resto.score}")
    print(f"[E2E] Semantic Match Score: {top_resto.score_breakdown.semantic_match}")
    
    # Verify semantic match is working (should be non-zero for this query)
    assert top_resto.score_breakdown.semantic_match > 0, "Semantic match score should be positive"
    
    # Verify LLM Explanation (Phase 3)
    # Note: This requires the GROQ_API_KEY to be set in environment
    if os.environ.get("GROQ_API_KEY"):
        assert top_resto.explanation is not None, "LLM Explanation should not be None when API key is provided"
        print(f"[E2E] AI Insight: {top_resto.explanation[:100]}...")
    else:
        print("[E2E] Skipping LLM check (GROQ_API_KEY not set in env)")

    print("[E2E] Full system check passed!")

if __name__ == "__main__":
    # Load .env manually for this script run
    from dotenv import load_dotenv
    load_dotenv("/Users/shrishpandey/Desktop/cursor_project/first-genai-project/restaurant_reco_phase5/.env")
    
    try:
        test_full_system_e2e()
    except Exception as e:
        print(f"[E2E] FAILED: {e}")
        exit(1)
