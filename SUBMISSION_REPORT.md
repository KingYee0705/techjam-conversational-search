# TechJam Shopping Agent — Technical Report

## System overview

The submission is a deterministic, local conversational product-search agent.
It does not call a generative LLM or external service during evaluation. The
entry point is `starter.agent:Agent`, and one turn follows five stages:

1. `starter/dialog.py` maintains per-session state, separates active,
   superseded, declined, and explicitly negative preferences, detects intent
   overrides, and selects a clarification field.
2. `starter/retrieval.py` builds an in-memory SQLite FTS5 index over the frozen
   catalog and combines current-message, active-constraint, category, profile,
   and strict-conjunction routes.
3. `starter/embedding_retrieval.py` optionally loads or builds normalized local
   MiniLM product vectors and retrieves semantic nearest neighbours for the
   current structured intent.
4. `starter/ranking.py` deterministically reranks the lexical/semantic candidate
   union using retrieval,
   message, constraint, route, profile, quality, budget, size, and exclusion
   evidence. Hard requirements are stronger than soft preferences, while
   incomplete catalog metadata is not treated as a known mismatch.
5. `starter/agent.py` coordinates the modules, caches repeated candidate pools,
   avoids repeating products within an intent, uses candidate evidence to
   retarget a follow-up after an explicit decline, and returns the exact Agent
   API response shape.

## Model, dependencies, cost, and token use

- Generative model: none.
- Optional local embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- Core dependencies: Python 3.10+ standard library and SQLite with FTS5.
- Optional semantic dependencies: `requirements-embedding.txt`.
- Runtime network access: not required after local model provisioning.
- API credentials: not required.
- Estimated model/API cost: USD 0.
- Reported prompt tokens: 0.
- Reported completion tokens: 0.

The semantic model is loaded with `local_files_only=True`; installing the
Python package alone is therefore insufficient on a clean offline host. The
model must be downloaded during setup or included in the final allowed bundle.
If the model, dependency, or vector cache is unavailable, the agent logs the
problem and continues with its complete lexical/strict path.

The optional `TECHJAM_CATALOG_PATH` environment variable can point to the
frozen catalog. When it is unset, the agent resolves `data/catalog.jsonl`
relative to the source bundle rather than relying on the process working
directory.

Embedding configuration is explicit:

| Variable | Meaning |
| --- | --- |
| `TECHJAM_ENABLE_EMBEDDINGS` | Set to `1` to activate semantic candidates. |
| `TECHJAM_EMBEDDING_MODEL` | Local MiniLM model directory. |
| `TECHJAM_EMBEDDING_CACHE` | Reusable product-vector cache directory. |
| `TECHJAM_SEMANTIC_SCORE_SCALE` | Cosine-to-retrieval calibration; tested default `0.75`. |

The scale is **not** a 75% final semantic weight. For a semantic-only candidate:

```text
retrieval_score = 0.75 * max(cosine_similarity, 0)
```

That value enters the ranker's 45% retrieval component. The semantic route
marker participates only inside the separate 5% route-evidence component.

## Reproduction

From the repository root:

```bash
python3 -m unittest
python3 -m evaluator.local_evaluator
```

The second command is the local equivalent of the official harness and writes
`results.json`. The catalog must first be downloaded and verified according to
the repository README.

For hybrid mode, first provision the local model as shown in the README, then
run:

```bash
TECHJAM_ENABLE_EMBEDDINGS=1 \
TECHJAM_EMBEDDING_MODEL="$PWD/models/all-MiniLM-L6-v2-1110a243" \
TECHJAM_EMBEDDING_CACHE="$PWD/data/.embedding_cache" \
TECHJAM_SEMANTIC_SCORE_SCALE=0.75 \
python3 -m evaluator.local_evaluator --output results-hybrid.json
```

## Public development validation

The implementation passes 151 deterministic unit/integration tests. On the
released 200-session development set, the lexical configuration produced:

| Scenario | Hit Rate@10 | MRR | MTTC |
| --- | ---: | ---: | ---: |
| Boundary | 1.000000 | 0.862500 | 3.600000 |
| Browsing | 1.000000 | 0.857827 | 2.637500 |
| Buying | 1.000000 | 0.853869 | 2.337500 |
| Intent Override | 1.000000 | 0.893651 | 3.966667 |
| **Overall** | **1.000000** | **0.861851** | **2.765000** |

Overall Efficiency was `0.823500` and TechnicalScore was `0.923255`.
These are public-development measurements, not a guarantee about the 800
private sessions. Calibration experiments retained the balanced strict-route
floor of `0.60`, rejected a classic per-constraint reciprocal-rank-fusion
variant after it reduced the public score, and bounded the purchase-popularity
prior rather than allowing unlimited rank movement.

The optional dense candidate route was evaluated separately:

| Configuration | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lexical/strict baseline | 1.000000 | 0.861851 | 2.765000 | 0.823500 | 0.923255 |
| Hybrid, semantic scale 0.75 | 1.000000 | 0.862060 | 2.765000 | 0.823500 | **0.923318** |
| Hybrid, semantic scale 1.00 | 1.000000 | 0.861149 | 2.770000 | 0.823000 | 0.922945 |

At scale `0.75`, only one of 200 public sessions changed: its target moved from
rank 8 to rank 6 on the same turn. At `1.00`, two sessions improved and four
regressed. The result supports semantic candidate expansion but also shows that
dense evidence should not overpower exact lexical constraints. These are
public-development measurements and should not be interpreted as proof of a
material hidden-set gain.

The multi-turn shortlist and popularity prior were also tested independently,
rather than accepted only because their combination scored well:

| Early turns | Popularity promotion | Hit Rate@10 | MRR | MTTC | TechnicalScore |
| --- | --- | ---: | ---: | ---: | ---: |
| 4 results | capped at 4 positions | 1.000000 | 0.861851 | 2.765000 | 0.923255 |
| 4 results | disabled | 1.000000 | 0.769736 | 2.765000 | 0.895621 |
| 10 results | capped at 4 positions | 1.000000 | 0.717938 | 2.405000 | 0.887281 |
| 10 results | disabled | 1.000000 | 0.613619 | 2.405000 | 0.855986 |

Returning ten immediately finds products sooner, while short disjoint batches
use later customer answers to improve the rank of the first valid hit. The
submitted policy follows the published TechnicalScore tradeoff and never
removes the strict or broad retrieval fallbacks. The popularity prior only
reorders an already relevance-selected batch and cannot promote an item more
than four original positions.

## Feasibility measurements

Development-machine measurements from the 50,000-product catalog were:

- Agent/index initialization: approximately 1.61 seconds.
- First response: approximately 0.26 seconds.
- Sampled mean response latency: approximately 0.20 seconds.
- Sampled p95 response latency: approximately 0.35 seconds.
- Agent-only peak resident memory: approximately 258 MB.
- Model/API cost and network latency: zero.

Embedding-enabled measurements on the same development machine were:

- MiniLM product-vector dimensions: `50,000 x 384` float32.
- One-time CPU cache build: approximately 576.6 seconds.
- Product-vector cache size: approximately 74 MB.
- Minimal local model assets: approximately 88 MB, excluding Python packages.
- Cached model/vector load: approximately 3.04 seconds.
- Mean query embedding plus exact similarity search: approximately 0.016 seconds.

The full dependency environment also includes PyTorch and is larger than the
model assets alone. The cache should be built before a live demonstration; its
catalog SHA-256 fingerprint prevents accidental reuse with a different catalog.

These values are feasibility measurements, not organizer-hardware guarantees.
They should be remeasured in the final clean submission environment.

## Reliability and fallback behavior

- Recommendation IDs are catalog `parent_asin` values and are deduplicated.
- Ranking and retrieval are deterministic, with ASIN tie-breaking.
- Missing price or product text does not crash ranking.
- Known excluded values are placed behind compliant alternatives, while
  products with missing evidence remain eligible.
- Superseded values from intent overrides are not mistaken for dislikes.
- Common no-preference and negative-preference paraphrases are normalized
  without polluting the active search query.
- Candidate-aware clarification activates only after an explicit decline and
  only when controlled color/material/feature evidence clears coverage,
  entropy, and hysteresis thresholds.
- A ranking failure falls back to normalized retrieval order.
- A semantic-model or semantic-query failure falls back to broad and strict
  BM25 candidates without changing the Agent API.
- Product embeddings are reused only when the catalog fingerprint, cache
  version, and stored row count are consistent.
- Missing catalog data and SQLite builds without FTS5 fail at startup with a
  clear diagnostic instead of silently producing an invalid run.

## Limitations

- The parser is primarily English and remains less flexible than a full
  semantic language model for unusual paraphrases, misspellings, or implicit
  preferences.
- Fixed embedding calibration produced only a marginal improvement on the
  public simulator. Buying-versus-Browsing adaptive weighting and a standalone
  semantic final-score sweep have not yet been validated.
- A cold semantic run must provision the local model and build the product
  cache. The approximately 9.6-minute development-machine build is unsuitable
  for a live video and may violate a strict cold-start budget if not prepared.
- Catalog metadata is incomplete. In particular, most products have no known
  price, so price constraints cannot safely be used as unconditional filters.
- Candidate-aware questioning is deliberately limited to controlled lexical
  evidence; unusual attributes still use the deterministic fallback order.
- Popularity is used only inside the relevance-selected recommendation batch.
  It cannot change batch membership and is capped at four positions, but it
  remains a disclosed generalization risk for rare products.
- SQLite FTS5 and the in-memory index have a measurable memory footprint.

## Team contributions

- **icebear0511:** catalog loading, FTS5 indexing, multi-route retrieval, and
  semantic candidate prototyping.
- **KingYee0705:** deterministic ranking and constraint-aware scoring.
- **yj0-0:** dialog state, clarification behavior, and intent overrides.
- **Alexis-alt-debug:** Agent API orchestration, caching, and module integration.
- **legoleekokpeng:** evaluation, reproducibility, Git workflow, and submission
  documentation.

## Demonstration

This transcript was reproduced with the lexical configuration from public
sample `public_0004`, runtime session ID `demo_public_0004`, and the frozen
catalog. The scenario is
`intent_override`; the hidden target is `B07C2XPZ6D`, *Emmalise Women's Basic
Casual Long Camisole Adjustable Strap Cami Layering Top*.

**Turn 1**

- User: “I'm looking for Tops & Tees Tanks & Camis. Long torso camisole for
  extra coverage with spagetti adjustable strap for perfect fit”
- Agent: “Here are some options. What matters most to you—such as material,
  color, fit, budget, or intended use?”
- `ask_attribute`: `other`
- Recommendations: `B07C2XPZ6D`, `B08GJ7RG8D`, `B09Z7DBCKQ`, `B08RDGX7VW`

**Turn 2**

- User: “For that, what matters is: polyester; Polyester,Cotton,Spandex.”
- Agent: “Here are some options. Is there one more must-have detail I should
  prioritize?”
- `ask_attribute`: `other`
- Recommendations: `B08TBFQR2J`, `B088KD56CJ`, `B09J8M2JC1`, `B09Z6KDYT1`

**Turn 3 — intent override**

- User: “Actually, ignore my earlier preference. What I need is: polyester.”
- Agent: “Here are some options. Is there one more must-have detail I should
  prioritize?”
- `ask_attribute`: `other`
- Recommendations: `B088FRY4ZT`, `B09Z7DBCKQ`, `B09Z6KDYT1`, `B0B298LLJN`

**Turn 4 — successful recommendation**

- User: “For that, what matters is: Quality soft cottonblend camisole, 95%
  cotton, 5% spandex, Neon Colors 57% cotton 38% polyester 5% spandex; Long
  torso camisole for extra coverage with spagetti adjustable strap for perfect
  fit.”
- Agent: “Here are some options. Is there a particular feature you want me to
  prioritize?”
- `ask_attribute`: `feature`
- Recommendations, best to worst: `B07C2XPZ6D`, `B08TBFQR2J`, `B09J8M2JC1`,
  `B09WJZRQ77`, `B075DZQD4H`, `B07SFM5N3G`, `B09PTT6QTH`, `B088KD56CJ`,
  `B07CPPNRBP`, `B08ZCFG1XQ`

The target appeared before the override, which the evaluator correctly did not
score for the new intent. It returned on turn 4 at rank 1, producing the first
valid scored hit after the turn-3 override.
