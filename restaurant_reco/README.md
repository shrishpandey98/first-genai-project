## Restaurant Recommendation (Phase 0/1)

This folder implements **Phase 0 and Phase 1** from `ARCHITECTURE.md`:

- **Phase 0**: project scaffolding + configuration conventions
- **Phase 1**: Hugging Face dataset ingestion + normalization/cleaning + reproducible artifacts

### Dataset
- Source: `ManikaSaini/zomato-restaurant-recommendation` ([link](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation))

### What Phase 1 produces
- `data/processed/restaurants.parquet`: cleaned restaurant records
- `docs/DATA_DICTIONARY.md`: schema + cleaning rules

### Quickstart

Create a virtualenv and install:

```bash
cd restaurant_reco
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Build cleaned artifacts:

```bash
python -m restaurant_reco.data_pipeline.build \
  --out data/processed/restaurants.parquet \
  --data-dict docs/DATA_DICTIONARY.md
```

Run tests:

```bash
pytest
```

### Configuration (Phase 0)
- **LLM (Grok) keys are not used in Phase 0/1**, but the convention is:
  - `XAI_API_KEY`
  - `GROK_MODEL`

