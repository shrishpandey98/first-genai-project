import streamlit as st
import os
import time
import pandas as pd
from typing import List, Optional

# Set page config FIRST
st.set_page_config(
    page_title="Flavorscape - AI Restaurant Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium aesthetics
st.markdown("""
    <style>
    /* Styling for the main header */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        color: #1E1E1E;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #FF8E53);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 1.2rem;
        color: #6C757D;
        margin-bottom: 2rem;
    }
    
    /* Restaurant Card Styling */
    .restaurant-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #f0f0f0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .restaurant-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    .restaurant-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2b2b2b;
        margin-bottom: 8px;
    }
    .restaurant-rating {
        background-color: #4CAF50;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .restaurant-meta {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }
    .restaurant-cuisines {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .cuisine-tag {
        background-color: #f4f6f8;
        color: #4a5568;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .restaurant-reason {
        background-color: #fff8f0;
        border-left: 4px solid #ff9f43;
        padding: 10px 15px;
        margin-top: 15px;
        border-radius: 0 8px 8px 0;
        font-size: 0.95rem;
        color: #2c3e50;
    }
    .ai-summary-box {
        background: linear-gradient(to right, #f8f9fa, #ffffff);
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 30px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .ai-summary-title {
        font-weight: 700;
        color: #343a40;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Imports from backend
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import sys
# Add src directory to sys.path so we can import modules
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, "src"))

try:
    from src.restaurant_reco_phase5.data_access import load_restaurants_parquet
    from src.restaurant_reco_phase5.models import RecommendationRequest, Budget, RecommendationResponse
    from src.restaurant_reco_phase5.recommender import recommend
except ImportError:
    from restaurant_reco_phase5.data_access import load_restaurants_parquet
    from restaurant_reco_phase5.models import RecommendationRequest, Budget, RecommendationResponse
    from restaurant_reco_phase5.recommender import recommend

# Load data with caching
@st.cache_data
def get_data() -> pd.DataFrame:
    # Try multiple possible locations for the data
    possible_paths = [
        os.environ.get("DATA_PATH"),
        "../restaurant_reco/data/processed/restaurants.parquet",
        "restaurant_reco/data/processed/restaurants.parquet",
        "data/processed/restaurants.parquet"
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            return load_restaurants_parquet(path)
            
    # If not found, attempt absolute relative paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fallback_path = os.path.join(base_dir, "..", "..", "restaurant_reco", "data", "processed", "restaurants.parquet")
    if os.path.exists(fallback_path):
        return load_restaurants_parquet(fallback_path)
        
    raise FileNotFoundError("Could not find the restaurants.parquet database. Please ensure the data ingestion phase was run.")

# Helper function to get available cuisines
@st.cache_data
def get_available_cuisines(df: pd.DataFrame) -> List[str]:
    # Flatten all cuisines
    all_cuisines = set()
    for row in df["cuisines"].dropna():
        if isinstance(row, str):
            for c in row.split(","):
                all_cuisines.add(c.strip())
        else:
            try:
                for c in row:
                    all_cuisines.add(str(c).strip())
            except TypeError:
                pass
    return sorted(list(all_cuisines))

# Helper function to get available locations
@st.cache_data
def get_available_locations(df: pd.DataFrame) -> List[str]:
    return sorted(df["location"].dropna().unique().tolist())


def main():
    st.markdown('<div class="main-header">🍽️ Flavorscape</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Discover your next favorite meal using AI-driven semantic recommendations.</div>', unsafe_allow_html=True)

    try:
        df = get_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
        
    locations = get_available_locations(df)
    cuisines_list = get_available_cuisines(df)
    
    # Sidebar
    st.sidebar.markdown("### 🎯 **Your Preferences**")
    
    # Needs to be a valid place from our data
    place = st.sidebar.selectbox("Neighborhood", [""] + locations, index=0)
    
    st.sidebar.markdown("---")
    
    cuisines = st.sidebar.multiselect("Cuisines", cuisines_list, placeholder="Select cuisines...")
    
    use_budget = st.sidebar.checkbox("Apply Budget Constraints?")
    budget_min, budget_max = 0, 2000
    if use_budget:
        cols = st.sidebar.columns(2)
        budget_min = cols[0].number_input("Min Cost (₹)", min_value=0, max_value=10000, step=100, value=0)
        budget_max = cols[1].number_input("Max Cost (₹)", min_value=0, max_value=10000, step=100, value=2000)
    
    min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 4.0, 0.1)
    
    st.sidebar.markdown("---")
    online_order = st.sidebar.checkbox("Must support Online Order", value=False)
    book_table = st.sidebar.checkbox("Must support Table Booking", value=False)
    
    top_n = st.sidebar.slider("Number of Recommendations", 1, 20, 5)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("🔧 **Advanced Engine Options**")
    debug_mode = st.sidebar.checkbox("Show Debug Stats")

    # Main area
    st.markdown("### 💬 Semantic Filters")
    free_text = st.text_input("Looking for something specific?", placeholder="e.g., 'A cozy Italian place with great tiramisu perfect for a romantic date'")
    
    find_button = st.button("✨ Give me Recommendations", type="primary", use_container_width=True)
    
    if find_button:
        if not place:
            st.error("Please select a Neighborhood from the sidebar!")
            return
            
        # Construct Request
        budget_obj = None
        if use_budget and (budget_min > 0 or budget_max > 0):
            budget_obj = Budget(
                min=budget_min if budget_min > 0 else None,
                max=budget_max if budget_max > 0 else None
            )
            
        req = RecommendationRequest(
            place=place,
            cuisines=cuisines if cuisines else [],
            min_rating=min_rating if min_rating > 0 else None,
            budget=budget_obj,
            online_order=online_order if online_order else None,
            book_table=book_table if book_table else None,
            top_n=top_n,
            free_text=free_text if free_text and free_text.strip() else None,
            debug=debug_mode
        )
        
        try:
            from src.restaurant_reco_phase5.scoring import ScoringWeights
        except ImportError:
            from restaurant_reco_phase5.scoring import ScoringWeights
            
        custom_weights = ScoringWeights()
        if req.free_text:
            custom_weights = ScoringWeights(
                semantic_match=0.40,
                rating=0.20,
                cuisine_match=0.10,
                votes=0.05,
                budget_fit=0.15,
                location_match=0.10
            )
        
        with st.spinner("🧠 Analyzing and finding the perfect spots..."):
            try:
                start_ts = time.time()
                resp = recommend(restaurants=df, request=req, weights=custom_weights)
                end_ts = time.time()
                
                # Show results
                if resp.summary:
                    st.markdown(f"""
                    <div class="ai-summary-box">
                        <div class="ai-summary-title">✨ AI Concierge Summary</div>
                        {resp.summary}
                    </div>
                    """, unsafe_allow_html=True)
                    
                if not resp.recommendations:
                    st.warning("No restaurants found matching those exact criteria. Try broadening your filters!")
                    
                st.markdown(f"**Found {len(resp.recommendations)} exceptional spots for you:**")
                
                for idx, r in enumerate(resp.recommendations, 1):
                    # Cuisine tags
                    c_tags = "".join([f'<span class="cuisine-tag">{c}</span>' for c in r.cuisines])
                    
                    # Construct card HTML
                    rating_display = f"⭐ {r.rating} ({r.votes} votes)" if r.rating else "No Rating"
                    cost_display = f"₹{r.cost_for_two} for two" if r.cost_for_two else "Cost info unavailable"
                    
                    amenities = []
                    if r.online_order: amenities.append("🟢 Online Delivery")
                    if r.book_table: amenities.append("🪑 Book Table")
                    amenities_display = " | ".join(amenities) if amenities else "No specific amenities"
                    
                    reasons_html = ""
                    if r.reasons:
                        reasons_list = "<br>".join([f"• {reason}" for reason in r.reasons])
                        reasons_html = f'<div class="restaurant-reason"><strong>Why we picked this:</strong><br>{reasons_list}</div>'
                        
                    card_html = f"""
                    <div class="restaurant-card">
                        <div style="display: flex; justify-content: space-between; align-items: start;">
                            <div>
                                <div class="restaurant-title">{idx}. {r.name}</div>
                                <div class="restaurant-meta">📍 {r.address}</div>
                                <div class="restaurant-meta">💵 {cost_display} | {amenities_display}</div>
                            </div>
                            <div class="restaurant-rating">{rating_display}</div>
                        </div>
                        <div class="restaurant-cuisines">
                            {c_tags}
                        </div>
                        {reasons_html}
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
                    
                    with st.expander("More details & external links"):
                        if r.url:
                            st.markdown(f"🔗 [View on Zomato]({r.url})")
                        if r.explanation:
                            st.write("**LLM Insight:**")
                            st.write(r.explanation)
                        if debug_mode and r.score_breakdown:
                            st.write("**Score Breakdown:**")
                            st.json(r.score_breakdown.dict())
                            
                if debug_mode and resp.debug:
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("### 🐛 Debug Stats")
                    st.sidebar.json(resp.debug.dict())
                    st.sidebar.caption(f"Latency: {(end_ts - start_ts)*1000:.0f} ms")
                    
            except Exception as e:
                st.error(f"❌ An error occurred generating recommendations: {str(e)}")


if __name__ == "__main__":
    main()
