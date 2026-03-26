## AI Restaurant Recommendation Service — Architecture (Phased)

### 1) Goal and scope
- **Goal**: Provide clear restaurant recommendations given user preferences (price, place, rating, cuisine), using the Zomato dataset from Hugging Face and **Grok (xAI)** for natural-language reasoning/summary.
- **Inputs**: location/place, budget/price range, minimum rating, cuisines, optional constraints (online order, book table), and optional free-text “vibes/requirements”.
- **Outputs**: ranked list of restaurants with a short explanation per item + “why it matches” + key fields (name, area, cuisines, rating, approx cost, link).
- **Dataset**: `ManikaSaini/zomato-restaurant-recommendation` on Hugging Face (CSV/Parquet) containing columns like `name`, `location`, `cuisines`, `rate`, `approx_cost(for two people)`, `votes`, `url`, `address`, etc. ([dataset page](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation))

### 2) Non-goals (initially)
- Real-time availability, delivery ETAs, table inventory, or live prices (dataset is static/historical).
- User accounts, long-term personalization, or “learning to rank” training loops (can be added later).
- Reliance on the LLM as the sole selector (LLM should *explain* and *summarize*, not be the only filter).

### 3) Guiding principles
- **Deterministic shortlist first**: Filter + score candidates using structured data, then use the LLM to generate clear recommendations and rationale.
- **Grounded responses**: LLM is constrained to the shortlisted items; every recommendation must map back to dataset rows.
- **Explainability**: Each recommendation includes explicit feature matches (budget, area, rating, cuisines).
- **Observability and evaluation built-in**: Track “why this was recommended”, latency, and user feedback.

---

## 4) High-level system overview

### Core modules
- **Client**: Web UI / CLI / Postman use (later). Sends preference request, receives recommendations.
- **API Service**: Validates input, orchestrates retrieval/ranking, calls LLM for final wording, returns response.
- **Data Pipeline (offline)**: Downloads dataset, cleans/normalizes fields, creates search index and/or embeddings.
- **Candidate Retrieval & Ranking**:
  - Structured filtering (place, min rating, budget range).
  - Text/cuisine matching (exact + fuzzy).
  - Optional semantic retrieval over reviews / cuisines / restaurant text.
  - Final scoring and top-K selection.
- **LLM Layer (Grok / xAI)**: Takes top-K structured candidates and generates user-facing recommendations (bullet list + reasons).
- **Store/Index**:
  - **Metadata store** for cleaned dataset (e.g., parquet/SQLite/Postgres).
  - **Search index** (keyword/filters) and optional **vector index** (semantic).
- **Monitoring & Analytics**: Logs, traces, dashboard; captures user feedback.

### Data flow (request time)
1. User submits preferences.
2. API validates and normalizes (e.g., “₹800 for two” → numeric).
3. Retrieve candidates:
   - Filter by location/place and numeric constraints.
   - Match cuisines; broaden if too few results.
4. Rank candidates using a transparent scoring function.
5. Send top-K candidates to the LLM with strict instructions to only use provided candidates.
6. Return final ranked recommendations with structured fields + LLM explanations.

---

## 5) Data model and normalization

### Canonical restaurant record (cleaned)
- **restaurant_id**: stable ID (e.g., hash of `name+address+location`).
- **name**
- **address**
- **location**: normalized area/neighborhood
- **city**: if applicable
- **cuisines**: list of strings
- **rating**: numeric float (parse from `rate` like `4.1/5`)
- **votes**: integer
- **cost_for_two**: numeric integer (parse `approx_cost(for two people)`)
- **online_order**: boolean
- **book_table**: boolean
- **url**
- **text_blob** (optional): concatenation of name + cuisines + rest_type + dish_liked + sample reviews (if used)

### Cleaning rules (offline)
- **Ratings**: handle missing values like `NEW`, `-`, or null; keep as `None` and exclude unless user allows.
- **Cost**: strip commas/currency; handle nulls.
- **Cuisines**: split by comma; trim; standardize casing (e.g., `North Indian`).
- **Location**: normalize synonyms and whitespace; keep original for display.
- **Deduplication**: de-duplicate by (name, address, location) heuristic and/or url.

---

## 6) Recommendation strategy (hybrid)

### Candidate generation
- **Hard filters** (when provided):
  - location/place match (exact → fuzzy fallback)
  - rating >= min_rating
  - cost_for_two within budget range
  - cuisine overlap >= 1 (or expand cuisines if too strict)
- **Soft filters**:
  - votes threshold (boost higher vote counts)
  - rest_type or online_order/book_table preferences

### Ranking (transparent score)
Compute a weighted score, for example:
- **rating_score** (normalized)
- **vote_score** (log-scaled)
- **budget_fit_score** (distance to target budget)
- **cuisine_match_score** (Jaccard overlap or weighted overlap)
- **location_match_score** (exact vs fuzzy)

Return top-K (e.g., K=10–20) to the LLM for explanation/formatting.

### LLM responsibilities (constrained)
- Convert top-K structured candidates into:
  - top-N recommendations (e.g., 5)
  - clear reasons tied to explicit fields
  - trade-offs (e.g., “slightly above budget but higher rating”)
  - optional next questions if results are weak (“increase budget or change area?”)

### LLM safeguards
- Provide candidates as JSON and instruct:
  - do not invent restaurants or fields
  - cite only provided restaurants
  - if insufficient results, say so and propose adjustments
- **Provider**: Use **Grok (xAI)** via an `llm` adapter, so provider changes don't touch recommender logic.
- **Config (example env vars)**:
  - `XAI_API_KEY`
  - `GROK_MODEL` (e.g., a Grok model identifier)
  - optional: `LLM_TEMPERATURE`, `LLM_TIMEOUT_MS`

---

## 7) API surface (planned)

### Endpoints
- `POST /recommendations`
  - **request**:
    - `place` (string)
    - `budget_for_two` (number | range)
    - `min_rating` (number)
    - `cuisines` (string[])
    - optional: `online_order`, `book_table`, `num_results`, `free_text`
  - **response**:
    - `recommendations`: array of `{name, location, cuisines, rating, cost_for_two, url, reasons[]}`
    - `debug` (optional): scoring breakdown, filters applied, candidate_count

### Contracts and validation
- Strong validation + defaults (e.g., min_rating default 3.8, num_results default 5).
- Deterministic behavior for same inputs unless explicitly randomized.

---

## 8) Storage and indexing options (choose per phase)

### Minimal (Phase 1–2)
- **Local Parquet/CSV** for cleaned dataset.
- **SQLite** for quick querying/filtering.

### Scalable (Phase 3+)
- **Postgres** for structured querying and metadata.
- **Search index** (optional): OpenSearch/Elasticsearch for keyword + filters.
- **Vector DB** (optional): for semantic search over reviews/text (FAISS locally or managed vector store).

---

## 9) Phases (delivery plan)

### Phase 0 — Project setup (foundation)
- Define product behavior: ranking rules, output format, fallback behavior.
- Add docs: `ARCHITECTURE.md`, `README.md`, basic runbooks.
- Define configuration strategy (env vars for **xAI/Grok** keys + model name, dataset version pinning).

### Phase 1 — Data ingestion + cleaning (offline)
- Pull dataset from Hugging Face.
- Build a reproducible cleaning pipeline:
  - normalize ratings/cost/cuisines/location
  - deduplicate
  - produce a **cleaned dataset artifact** (parquet) and a data dictionary.
- Add dataset versioning and checksums to ensure reproducibility.

### Phase 2 — Deterministic recommender (no LLM dependency)
- Implement filter + rank + return top-N in structured JSON.
- Add a simple baseline scoring function and debug outputs.
- Add basic evaluation with a small set of test queries (golden cases).

### Phase 3 — LLM-assisted explanations (grounded)
- Add LLM call that takes top-K candidates and writes:
  - short explanations, pros/cons, and final formatting.
- Add prompt guardrails:
  - “use only provided candidates”
  - refusal/insufficient results behavior
- Add caching for LLM outputs keyed by (query + candidate ids).

### Phase 4 — Retrieval improvements (semantic + robustness)
- Add semantic retrieval over `text_blob` (cuisines + reviews) to improve matching when user uses free-text.
- Add query expansion for cuisines (“South Indian” ↔ “Udupi”, etc.) and location fuzzy matching.
- Add reranking (optional): smaller LLM or cross-encoder reranker on shortlist.

### Phase 5 — Production readiness
- Deploy API service (containerized) and choose storage/index based on scale.
- Add observability (structured logs, tracing, metrics):
  - latency breakdown (retrieve/rank/LLM)
  - top features influencing ranking
  - error rates and LLM failures
- Add safety + compliance:
  - no secrets in logs
  - PII policy (requests likely minimal; still treat carefully)

### Phase 6 — Quality loop and personalization (optional)
- Collect user feedback (“helpful” / “not helpful”, clicked restaurant).
- Offline evaluation dashboards; adjust weights and heuristics.
- Optional user profiles and session-based preferences.

### Phase 7 — Web UI (user-facing product surface)
- Build a **Web UI** where users enter preferences and view recommendations.
- **Primary flows**:
  - Preference form: `place`, `budget_for_two`, `min_rating`, `cuisines` (+ toggles like online order / book table)
  - Results page: ranked cards, “why this matches”, key fields (rating, cost, cuisines), and outbound `url`
  - Feedback capture: thumbs up/down + optional “what went wrong?” (used by Phase 6)
  - Empty/low-results handling: suggest relaxing filters (budget/rating/location/cuisines)
- **Frontend ↔ API integration**:
  - Calls `POST /recommendations` and renders structured response
  - Optional “debug mode” (hidden behind a toggle) to show scoring breakdown for dev/admin
- **Web UI architecture options** (choose later):
  - Single-page app (React/Next.js) calling the API
  - Server-rendered UI (Next.js/Remix) proxying API calls
- **Security/ops**:
  - Rate limit at the edge (and/or API)
  - If auth is added later: simple API key/session for internal use; keep public mode anonymous
  - Basic analytics: page load + submit + latency + click-through + feedback events

---

## 10) Cross-cutting concerns

### Security
- API keys in environment variables / secret manager.
- Rate limits and abuse protection.

### Reliability
- Timeouts and fallbacks:
  - if **Grok** fails/timeouts → return deterministic top-N with templated explanations
- Circuit breaker for **xAI/Grok**.

### Testing strategy
- **Unit tests**: parsing, normalization, scoring.
- **Golden tests**: fixed inputs → stable ranked outputs.
- **Prompt tests**: ensure LLM does not hallucinate; verify all outputs reference provided candidates.

### Observability
- Request IDs; structured logs with:
  - applied filters, candidate count
  - top-5 scoring contributions
  - **Grok** model name, token usage (if available), and latency

---

## 11) Suggested repository structure (when you implement)
- `docs/`
  - `ARCHITECTURE.md` (this file)
  - `DATA_DICTIONARY.md`
  - `EVAL.md`
- `data/`
  - `raw/` (ignored in git if large)
  - `processed/` (artifacts)
- `src/`
  - `api/` (request handling, validation)
  - `recommender/` (filters, scoring, ranking)
  - `llm/` (prompting, providers, caching)
  - `data_pipeline/` (ingest, clean, build index)
  - `common/` (config, logging, types)
- `tests/`
  - `unit/`
  - `golden/`

