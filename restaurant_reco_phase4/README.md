## Phase 3 — LLM-Enriched Restaurant Recommender

This package implements **Phase 3** from `ARCHITECTURE.md`: a deterministic, explainable recommender that uses an LLM to generate explanations and summaries:

- reads the **Phase 1 cleaned Parquet** (canonical schema)
- validates user preferences (place, budget, min rating, cuisines)
- filters candidates deterministically
- ranks using a transparent scoring function
- uses the **Groq API** to generate summaries and reasons for the recommendations.
- returns a structured JSON response (optionally with debug breakdown)

### Install (editable)

```bash
cd restaurant_reco_phase4
python3 -m pip install -e ".[dev]"
```

### Run (CLI)

Ensure you have your Groq API key exported:
```bash
export GROQ_API_KEY="your-api-key"
```

```bash
python3 -m restaurant_reco_phase4.cli \
  --data ../restaurant_reco/data/processed/restaurants.parquet \
  --place "Banashankari" \
  --budget 800 \
  --min-rating 4.0 \
  --cuisine "North Indian" \
  --top-n 5 \
  --debug
```

