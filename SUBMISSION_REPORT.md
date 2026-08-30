# TechJam Shopping Agent — Technical Report

## System overview

The submission is a deterministic, offline conversational product-search agent.
It does not call an LLM or any external service during evaluation. The entry
point is `starter.agent:Agent` and the implementation has four stages:

1. `starter/dialog.py` maintains per-session state, separates active,
   superseded, declined, and explicitly negative preferences, detects intent
   overrides, and selects a clarification field.
2. `starter/retrieval.py` builds an in-memory SQLite FTS5 index over the frozen
   catalog and combines current-message, active-constraint, category, profile,
   and strict-conjunction routes.
3. `starter/ranking.py` deterministically reranks candidates using retrieval,
   message, constraint, route, profile, quality, budget, size, and exclusion
   evidence. Hard requirements are stronger than soft preferences, while
   incomplete catalog metadata is not treated as a known mismatch.
4. `starter/agent.py` coordinates the modules, caches repeated candidate pools,
   avoids repeating products within an intent, uses candidate evidence to
   retarget a follow-up after an explicit decline, and returns the exact Agent
   API response shape.

## Model, dependencies, cost, and token use

- Model: no generative model; deterministic lexical retrieval and ranking.
- Runtime dependencies: Python 3.10+ standard library and SQLite with FTS5.
- Network access: not required.
- API credentials: not required.
- Estimated model/API cost: USD 0.
- Reported prompt tokens: 0.
- Reported completion tokens: 0.

The optional `TECHJAM_CATALOG_PATH` environment variable can point to the
frozen catalog. When it is unset, the agent resolves `data/catalog.jsonl`
relative to the source bundle rather than relying on the process working
directory.

## Reproduction

From the repository root:

```bash
python3 -m unittest
python3 -m evaluator.local_evaluator
```

The second command is the local equivalent of the official harness and writes
`results.json`. The catalog must first be downloaded and verified according to
the repository README.

## Public development validation

The frozen implementation passes 146 deterministic unit/integration tests. On
the released 200-session development set it produced:

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
- Missing catalog data and SQLite builds without FTS5 fail at startup with a
  clear diagnostic instead of silently producing an invalid run.

## Limitations

- The parser is primarily English and remains less flexible than a full
  semantic language model for unusual paraphrases, misspellings, or implicit
  preferences.
- Catalog metadata is incomplete. In particular, most products have no known
  price, so price constraints cannot safely be used as unconditional filters.
- Candidate-aware questioning is deliberately limited to controlled lexical
  evidence; unusual attributes still use the deterministic fallback order.
- Popularity is used only inside the relevance-selected recommendation batch.
  It cannot change batch membership and is capped at four positions, but it
  remains a disclosed generalization risk for rare products.
- SQLite FTS5 and the in-memory index have a measurable memory footprint.

## Team contributions

- Person 1 — catalog loading, multi-route retrieval, and FTS indexing.
- Person 2 — standalone deterministic ranking and constraint-aware scoring.
- Person 3 — dialogue state, clarification behavior, and intent overrides.
- Person 4 — Agent API orchestration, caching, and module integration.
- Person 5 — evaluation, reproducibility, Git workflow, and submission report.

The team should replace the role labels with participant names in the final
submission form if the organizer requires named attribution.

## Demonstration

This transcript was reproduced from public sample `public_0004` using runtime
session ID `demo_public_0004` and the current frozen catalog. The scenario is
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
