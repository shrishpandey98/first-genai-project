from __future__ import annotations

import argparse
import json

from restaurant_reco_phase4.data_access import load_restaurants_parquet
from restaurant_reco_phase4.models import Budget, RecommendationRequest
from restaurant_reco_phase4.recommender import recommend


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Phase 3 LLM enriched restaurant recommender (JSON output).")
    parser.add_argument("--data", required=True, help="Path to Phase 1 processed restaurants parquet.")
    parser.add_argument("--place", required=True, help="Place/location (e.g., Banashankari).")
    parser.add_argument("--budget", type=int, default=None, help="Budget max for two people.")
    parser.add_argument("--min-rating", type=float, default=None, help="Minimum rating.")
    parser.add_argument("--cuisine", action="append", default=[], help="Preferred cuisine (repeatable).")
    parser.add_argument("--free-text", type=str, default=None, help="Any free text string to search via semantic matching.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of recommendations to return.")
    parser.add_argument("--online-order", action="store_true", help="Require online order support.")
    parser.add_argument("--book-table", action="store_true", help="Require table booking support.")
    parser.add_argument("--debug", action="store_true", help="Include debug info and score breakdown.")
    args = parser.parse_args()

    df = load_restaurants_parquet(args.data)

    budget = Budget(max=args.budget) if args.budget is not None else None
    req = RecommendationRequest(
        place=args.place,
        cuisines=args.cuisine,
        min_rating=args.min_rating,
        budget=budget,
        online_order=True if args.online_order else None,
        book_table=True if args.book_table else None,
        top_n=args.top_n,
        free_text=args.free_text,
        debug=args.debug,
    )

    resp = recommend(restaurants=df, request=req)
    print(json.dumps(resp.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

