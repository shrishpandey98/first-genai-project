## Phase 2 — Deterministic Restaurant Recommender

This package implements **Phase 2** from `ARCHITECTURE.md`: a deterministic, explainable recommender that:

- reads the **Phase 1 cleaned Parquet** (canonical schema)
- validates user preferences (place, budget, min rating, cuisines)
- filters candidates deterministically
- ranks using a transparent scoring function
- returns a structured JSON response (optionally with debug breakdown)

### Install (editable)

```bash
cd restaurant_reco_phase2
python3 -m pip install -e ".[dev]"
```

### Run (CLI)

```bash
python3 -m restaurant_reco_phase2.cli \
  --data ../restaurant_reco/data/processed/restaurants.parquet \
  --place "Banashankari" \
  --budget 800 \
  --min-rating 4.0 \
  --cuisine "North Indian" \
  --top-n 5 \
  --debug
```

