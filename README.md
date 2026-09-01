# Conversational Product Search

> A local, hybrid product-search agent that turns a multi-turn conversation
> into ranked recommendations from a frozen catalog of 50,000 Amazon products.

Customers rarely describe the right product in one perfect query. They start
vaguely, add constraints later, reject suggestions, or change their minds. Our
agent handles that conversation explicitly: it remembers the current intent,
asks one useful follow-up question at a time, and combines lexical search with
MiniLM embeddings before producing deterministic ranked recommendations.

## At a glance

| | |
| --- | --- |
| **Catalog** | 50,000 Clothing, Shoes, and Jewelry products |
| **Conversation** | Buying, Browsing, Intent Override, and Boundary sessions |
| **Retrieval** | Broad BM25 + strict all-concept search + MiniLM embeddings |
| **Ranking** | Constraint-aware, explainable, deterministic scoring |
| **Runtime** | Local Python; no generative LLM or external API during evaluation |
| **Public result** | Hit Rate@10 `1.0`; hybrid TechnicalScore `0.923318` |

These results are from the released 200-session public development set. They
are not a guarantee of performance on the organizer's 800 hidden sessions.

**Featured submission configuration:** hybrid retrieval with
`TECHJAM_ENABLE_EMBEDDINGS=1` and semantic scale `0.75`. The flag is explicit
so the result is reproducible and the same code can retain its zero-dependency
lexical fallback. Running without the hybrid environment variables reproduces
the lexical configuration rather than the `0.923318` hybrid result.

## Why this approach

A single keyword query loses the most useful part of conversational shopping:
information revealed over time. A single embedding search has the opposite
risk—it can retrieve semantically related products while weakening an exact
brand, color, material, or budget requirement.

We therefore separate the problem into four responsibilities:

1. **Understand the conversation.** Preserve active requirements, explicit
   negatives, declined attributes, and intent overrides across turns.
2. **Maximize candidate recall.** Search independent broad, strict, and
   semantic routes so one weak route cannot erase another route's candidates.
3. **Rank with current intent first.** Give the consolidated intent query and active
   constraints more influence than historical profile tags.
4. **Stay reproducible.** Use deterministic parsing, ranking, tie-breaking,
   and a complete lexical fallback.

## Architecture

```text
Customer message + anonymized profile
                  |
                  v
       Dialog state and intent memory
                  |
        +---------+----------+
        |         |          |
        v         v          v
   Broad BM25  Strict AND  MiniLM semantic
     routes      route      route (optional)
        |         |          |
        +---------+----------+
                  |
                  v
       Candidate union + deduplication
                  |
                  v
       Constraint-aware deterministic ranker
                  |
                  v
   Follow-up question + ranked `parent_asin` list
```

The embedding route is a **candidate generator**, not a replacement for BM25.
If embeddings are disabled or unavailable, broad and strict lexical retrieval
continue to provide a complete working agent.

## How one turn works

The public entry point is `starter.agent:Agent`.

1. `Agent.respond()` validates the turn and delegates message interpretation.
2. `DialogStateManager` updates active, negative, superseded, and declined
   preferences and detects an intent change.
3. `CatalogRetriever` searches the consolidated intent through broad BM25 and
   strict all-concept routes.
4. When enabled, `EmbeddingRetriever` adds semantic nearest neighbours from a
   cached MiniLM product matrix.
5. The agent unions the candidate pools and hydrates semantic ASINs with their
   full catalog records.
6. `rank_products()` scores message, constraint, route, profile, quality, and
   budget evidence without treating missing metadata as a known mismatch.
7. The agent selects unseen products, chooses the next clarification field,
   and returns the exact competition API response.

### Ranking signals

| Signal | Base weight | Purpose |
| --- | ---: | --- |
| Retrieval evidence | 45% | Strength of BM25, strict, or calibrated semantic retrieval |
| Current-intent query | 25% | Token and phrase agreement with the consolidated request |
| Active constraints | 20% | Match against the structured multi-turn intent |
| Route evidence | 5% | Agreement across independent candidate routes |
| Profile tags | 3% | Small personalization boost without overriding current intent |
| Quality prior | 2% | Rating and popularity as a bounded tie-breaker |

Hard requirements receive additional bounded influence. Known exclusions are
placed behind compliant alternatives, while missing price or attribute data is
treated as unknown rather than automatically wrong.

## Embedding retrieval

The featured semantic route uses
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
to produce normalized 384-dimensional vectors.

### When embeddings are computed

- **Product embeddings:** computed once when semantic mode starts and no valid
  cache exists. The cache is fingerprinted with the catalog SHA-256 digest.
- **Query embedding:** computed for each new dialog intent from the latest
  message, category, and active constraints.
- **Similarity:** calculated by multiplying the normalized product matrix by
  the normalized query vector. The dot product is therefore cosine similarity.
- **Repeated state:** an in-memory candidate cache avoids doing the same search
  again for an unchanged dialog state.

Positive cosine similarity is conservatively calibrated before it enters the
ranker:

```text
semantic retrieval_score = 0.75 × max(cosine_similarity, 0)
```

The `0.75` value is a retrieval-score calibration—not a 75% final semantic
weight. If an ASIN is found through multiple routes, the ranker retains its
strongest retrieval score and merges the route labels.

## Public development results

| Configuration | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| Weak starter | 0.125000 | 0.068034 | 9.810 | 0.1190 | 0.106710 |
| Our lexical/strict agent | 1.000000 | 0.861851 | 2.765 | 0.8235 | 0.923255 |
| **Hybrid, semantic scale 0.75** | **1.000000** | **0.862060** | **2.765** | **0.8235** | **0.923318** |
| Hybrid, semantic scale 1.00 | 1.000000 | 0.861149 | 2.770 | 0.8230 | 0.922945 |

The conservative semantic route changed only one public session, moving its
target from rank 8 to rank 6 on the same turn. Increasing semantic influence
caused more regressions than improvements, so we retained the `0.75` setting.
This small but measurable result is why we describe the system as
**hybrid-ready and lexical-first**, rather than claiming embeddings solve every
query.

The competition metric is:

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

See [SUBMISSION_REPORT.md](SUBMISSION_REPORT.md) for scenario-level results,
ablation studies, feasibility measurements, and limitations.

## Quick start

### Requirements

- Python 3.10 or later
- SQLite compiled with FTS5 support
- The released frozen catalog
- Optional for semantic mode: NumPy, Sentence Transformers, and a locally
  provisioned MiniLM model

### 1. Download and verify the catalog

Download `catalog.jsonl.gz` and `SHA256SUMS` from the GitHub Release attached to
the challenge repository. From the repository root:

```bash
shasum -a 256 -c SHA256SUMS
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Linux users may use `sha256sum -c SHA256SUMS` for the first command.

### 2. Run the deterministic lexical agent

```bash
python3 -m unittest
python3 -m evaluator.local_evaluator --output results-lexical.json
```

The evaluator writes aggregate and per-session output to `results.json`.

### 3. Run the featured hybrid agent

Install the optional dependencies and download the model while network access
is available:

```bash
python3 -m pip install -r requirements-embedding.txt
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41').save('models/all-MiniLM-L6-v2-1110a243')"
shasum -a 256 models/all-MiniLM-L6-v2-1110a243/model.safetensors
```

The expected `model.safetensors` SHA-256 is
`53aa51172d142c89d9012cce15ae4cec7409935868ccecf3954370f96732a1`.

Then run locally:

```bash
TECHJAM_ENABLE_EMBEDDINGS=1 \
TECHJAM_EMBEDDING_MODEL="$PWD/models/all-MiniLM-L6-v2-1110a243" \
TECHJAM_EMBEDDING_CACHE="$PWD/data/.embedding_cache" \
TECHJAM_SEMANTIC_SCORE_SCALE=0.75 \
python3 -m evaluator.local_evaluator --output results-hybrid.json
```

The first semantic startup builds the catalog-vector cache. Later runs reuse
it. The model loader uses `local_files_only=True`, so an offline evaluation
machine must be provisioned in advance.

### Environment variables

| Variable | Required | Meaning |
| --- | --- | --- |
| `TECHJAM_CATALOG_PATH` | No | Used when constructing `Agent()` directly; evaluator users should pass `--catalog` |
| `TECHJAM_ENABLE_EMBEDDINGS` | Hybrid only | Set to `1` to enable semantic candidates |
| `TECHJAM_EMBEDDING_MODEL` | Recommended | Pinned local model path; code otherwise uses its local-only default |
| `TECHJAM_EMBEDDING_CACHE` | No | Optional directory for reusable product vectors |
| `TECHJAM_SEMANTIC_SCORE_SCALE` | No | Semantic retrieval calibration; tested value `0.75` |

## Agent API

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [{"parent_asin": "B000..."}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
```

Only catalog-valid, unique `parent_asin` values are returned. Recommendation
order determines the score.

## Repository guide

| Path | Responsibility |
| --- | --- |
| `starter/agent.py` | Agent API, orchestration, caching, candidate union, response policy |
| `starter/dialog.py` | Multi-turn state, declines, exclusions, clarification, intent overrides |
| `starter/retrieval.py` | In-memory FTS5 index, broad BM25 routes, strict retrieval |
| `starter/embedding_retrieval.py` | Optional cached MiniLM candidate route |
| `starter/ranking.py` | Deterministic constraint-aware fusion and ranking |
| `evaluator/local_evaluator.py` | Official-style public simulator and scorer |
| `tests/` | 151 deterministic unit and integration tests |
| `SUBMISSION_REPORT.md` | Technical method, experiments, feasibility, and limitations |

For a code walkthrough, follow one request instead of reading files in
alphabetical order:

```text
Agent.respond
  → dialog state
  → broad/strict BM25 + semantic retrieval
  → candidate union
  → deterministic ranking
  → response selection
  → evaluator results
```

## Feasibility and cost

Development-machine measurements on the 50,000-product catalog:

| Measurement | Lexical | Embedding-enabled |
| --- | ---: | ---: |
| Lexical index initialization | ~1.61 s | Also required in hybrid mode |
| Cached MiniLM/vector load | — | ~3.04 s |
| Mean response latency | ~0.20 s | +~0.016 s query embedding/search |
| p95 response latency | ~0.35 s | Not separately standardized |
| Agent peak resident memory | ~258 MB | Plus ~74 MB product-vector cache |
| One-time embedding cache build | — | ~576.6 s on CPU |
| External API/model cost | USD 0 | USD 0 |
| Prompt/completion tokens | 0 / 0 | 0 / 0 |

Measurements depend on hardware and should be repeated in the final clean
submission environment.

The reported hybrid experiment used an Apple M3 MacBook Air with 16 GB RAM,
macOS 26.4.1, Python 3.13.6, SQLite 3.50.4/FTS5, NumPy 2.5.2,
Sentence Transformers 5.1.0, Transformers 4.57.6, and PyTorch 2.13.0 on CPU.
The 74 MB figure is the vector artifact size; full hybrid peak RSS was not
measured.

## Reliability and limitations

- The parser is primarily English and may be weaker on unusual paraphrases,
  typos, or implicit preferences.
- Catalog metadata is incomplete, especially price, so unknown values cannot
  safely be treated as violations.
- The semantic gain on the public set is marginal; it should be treated as a
  recall hedge, not proof of hidden-set improvement.
- A cold embedding run requires a locally provisioned model and a one-time
  cache build. Prebuild it before recording a live demonstration.
- Popularity can reorder only the already relevance-selected batch and is
  capped, but it remains a disclosed risk for rare products.
- If ranking fails, the agent uses deterministic retrieval order. If semantic
  retrieval fails, it continues with broad and strict BM25 without changing
  the API contract.

## Team contributions

- **Person 1:** catalog loading, FTS5 indexing, multi-route retrieval, and
  semantic candidate prototyping.
- **Person 2:** deterministic ranking and constraint-aware scoring.
- **Person 3:** dialog state, clarification behavior, and intent overrides.
- **Person 4:** Agent API orchestration, caching, and module integration.
- **Person 5:** evaluation, reproducibility, Git workflow, and submission
  documentation.

Replace the role labels with participant names before the final Devpost
submission if named attribution is required.

## Data attribution

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab,
UCSD. See [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md) before using or
redistributing the data. Raw user IDs, review text, timestamps, and purchase
history are not exposed to the agent.
