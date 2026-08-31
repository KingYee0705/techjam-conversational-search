# Judge Demo Video Walkthrough

This is the recommended recording order and spoken script for the hybrid
conversational shopping agent. It is written for a **6–7 minute** technical
walkthrough. The words under **Say** can be read almost verbatim; the words
under **Show** are screen directions.

## The correct repository sequence

Do not present the repository as four complete files in a straight line:

```text
agent.py -> dialog.py -> retrieval.py -> ranking.py
```

That order hides the parallel semantic route and the points where control
returns to the orchestrator. Instead, trace one call to `Agent.respond`:

```text
Customer turn
     |
     v
Agent.respond                         starter/agent.py
     |
     v
Structured dialog decision           starter/dialog.py
     |
     +-------------------+-----------------------+
     |                   |                       |
     v                   v                       v
Broad BM25 routes   Strict AND route       Semantic route
starter/retrieval.py                     starter/embedding_retrieval.py
     |                   |                       |
     +-------------------+-----------------------+
                         |
                         v
               Candidate union in Agent
                         |
                         v
               Constraint-aware ranking          starter/ranking.py
                         |
                         v
            Unseen shortlist + API response       starter/agent.py
                         |
                         v
                Deterministic evaluator           evaluator/local_evaluator.py
```

Use this file/function sequence:

1. `README.md` — challenge, architecture, and headline metrics.
2. `starter/agent.py` — `Agent.reset` and `Agent.respond`.
3. `starter/dialog.py` — `DialogStateManager.process_turn` and `_decision`.
4. Return to `starter/agent.py` — `_retrieve_candidates`.
5. `starter/retrieval.py` — `retrieve_products` and
   `retrieve_strict_products`.
6. `starter/embedding_retrieval.py` — `_build_cache` and `retrieve`.
7. Return to `starter/agent.py` — semantic hydration and candidate union.
8. `starter/ranking.py` — `SCORE_WEIGHTS` and `rank_products`.
9. Return to `starter/agent.py` — `_select_recommendations` and API output.
10. `evaluator/local_evaluator.py` — `evaluate`, followed by the result table.

This order makes the architecture easy to follow while avoiding a long scroll
through parsing helpers and regular expressions.

## Before recording

### Prepare the environment

Run the fast validation before recording:

```bash
python3 -m unittest
```

The expected result for this branch is:

```text
Ran 151 tests
OK
```

For the embedding-enabled demonstration, install the optional dependencies and
make the pretrained model available locally before the video:

```bash
python3 -m pip install -r requirements-embedding.txt
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('models/all-MiniLM-L6-v2')"
```

The second command needs network access only while preparing the local model.
The submitted runtime uses `local_files_only=True`, so the model path must be
present locally when the network is unavailable.

Enable the hybrid route with:

```bash
export TECHJAM_ENABLE_EMBEDDINGS=1
export TECHJAM_EMBEDDING_MODEL="$PWD/models/all-MiniLM-L6-v2"
export TECHJAM_EMBEDDING_CACHE="$PWD/data/.embedding_cache"
export TECHJAM_SEMANTIC_SCORE_SCALE=0.75
```

Prebuild the embedding cache before pressing Record. Do **not** make judges
watch the one-time catalog encoding step. Explain it, then show the completed
cache measurement.

### Prepare the editor

Open these tabs in advance:

- `README.md`
- `starter/agent.py`
- `starter/dialog.py`
- `starter/retrieval.py`
- `starter/embedding_retrieval.py`
- `starter/ranking.py`
- `evaluator/local_evaluator.py`
- `SUBMISSION_REPORT.md`

Increase the editor font size and use symbol search to jump directly to the
functions listed above. Avoid scrolling through every helper or regex.

## Detailed 6–7 minute narration

### 0:00–0:25 — Hook: shopping intent evolves

**Show:** A title slide with the architecture diagram, then a short example:
“I need shoes for standing all day” → “cushioned walking sneaker with arch
support.”

**Say:**

> Shopping intent rarely arrives as one perfect keyword query. A customer may
> begin vaguely, reveal a must-have later, reject an attribute, or completely
> change their mind. Our system is a local conversational shopping agent that
> remembers those changes and searches 50,000 products using precise BM25
> retrieval together with semantic MiniLM embeddings.

### 0:25–0:50 — Define the objective

**Show:** `README.md`, highlighting the ten-turn limit and exact-ASIN scoring.

**Say:**

> The challenge gives our agent at most ten turns to return the exact hidden
> `parent_asin` in its Top 10. The agent sees a short message and a privacy-safe
> aggregate preference profile—never raw reviews or purchase history. It may
> ask one structured follow-up attribute and recommend products on the same
> turn.

### 0:50–1:35 — Start at the public API

**Show:** `starter/agent.py`. Jump to `class Agent`, then `reset` and `respond`.

**Say:**

> `Agent` is the only public entry point. `reset` creates isolated session
> memory. `respond` is the traffic controller for one customer turn.
>
> First, it makes repeated evaluator calls idempotent. Next, it asks the dialog
> manager to interpret the latest message. If the customer overrides their
> intent, products shown under the previous intent become eligible again.
>
> The agent then generates a hybrid candidate pool, reranks it, selects unseen
> recommendations, and returns exactly four API fields: the natural-language
> message, one `ask_attribute`, catalog ASINs, and token usage.

**Point out:** The numbered comments inside `respond`. Do not explain every
helper function yet.

### 1:35–2:25 — Dialog state and intent override

**Show:** `starter/dialog.py`. Jump to `DialogStateManager.reset`,
`process_turn`, `_search_query`, and `_decision`.

**Say:**

> The dialog manager gives the system memory without using a generative API.
> It stores active preferences, explicit negatives, superseded values,
> declined attributes, the pending question, and conversation history as
> separate concepts.
>
> The pending question matters because a short answer such as “blue” should be
> understood as a color only if color was just requested. Interpretation has a
> deliberate precedence: a phrase such as “I don't mind the color” is handled
> as a decline before it can pollute active constraints, while “actually,
> ignore my earlier preference” replaces old intent instead of appending it.
>
> `_decision` is the clean handoff to search. It returns a consolidated query,
> category, active and negative constraints, hard-or-soft priority metadata,
> and the next allowed clarification attribute.

### 2:25–3:10 — Lexical candidate routes

**Show:** `starter/retrieval.py`. Jump to `_build_index`, `retrieve_products`,
and `retrieve_strict_products`.

**Say:**

> At startup, the lexical retriever loads the frozen catalog into an in-memory
> SQLite FTS5 index. BM25 is field weighted so title and category evidence is
> stronger than a long description.
>
> We search four views of the current intent independently: the current
> consolidated message, active constraints, category, and low-weight profile
> evidence. Normalized route scores are merged, so agreement between routes is
> useful evidence.
>
> A separate strict route requires every disclosed searchable concept. The
> broad routes protect recall; the strict route protects precision. They are
> unioned, so an empty strict result can never delete valid broad candidates.

### 3:10–4:10 — Semantic embeddings

**Show:** `starter/embedding_retrieval.py`. Highlight `DEFAULT_MODEL`,
`product_document`, `_build_cache`, and `retrieve`.

**Say:**

> Keyword search is strong for exact constraints such as black, leather, size
> eight, and under eighty dollars. It is weaker when user and product describe
> the same need with different words. Our optional dense route uses the local
> `sentence-transformers/all-MiniLM-L6-v2` model and normalized 384-dimensional
> vectors.
>
> There are two different computation times. On a cache miss, `_build_cache`
> encodes each product's title, category, store, features, details, and
> description once. The cache includes a catalog fingerprint, so embeddings
> from a different catalog cannot be silently reused.
>
> At query time, only the current structured intent is encoded. The line
> `scores = self.embeddings @ vector` compares that query against every cached
> product. Because both sides are normalized, the dot product is cosine
> similarity. We then take the closest product IDs.
>
> On our development machine, the one-time CPU build took about nine minutes
> and thirty-seven seconds and created a 74-megabyte vector cache. A cached
> model/index load took about three seconds, and one semantic query plus matrix
> search averaged about sixteen milliseconds.

### 4:10–4:40 — Explain the hybrid union accurately

**Show:** Return to `starter/agent.py::_retrieve_candidates`, starting at
`dense_query` and ending at the candidate union.

**Say:**

> This is candidate-union hybrid retrieval, not a claimed eighty-twenty score
> blend. We take broad BM25 candidates, strict candidates, and semantic nearest
> neighbours, then pass their union downstream.
>
> A semantic candidate's positive cosine similarity is multiplied by a
> calibrated scale of 0.75 before becoming retrieval evidence. That 0.75 is
> **not** a 75-percent final semantic weight. If the local model is missing or
> semantic search fails, the agent logs the issue and continues with BM25.

### 4:40–5:30 — Final deterministic ranking

**Show:** `starter/ranking.py`. Highlight `FIELD_WEIGHTS`, `ROUTE_WEIGHTS`,
`SCORE_WEIGHTS`, `_deduplicate`, and `rank_products`.

**Say:**

> The ranker is source-agnostic: every candidate follows the same contract.
> Duplicate ASINs keep their strongest retrieval score while their route names
> are combined.
>
> The final interpretable formula uses 45 percent retrieval evidence, 25
> percent current-message coverage, 20 percent active constraints, 5 percent
> route evidence, 3 percent profile overlap, and 2 percent rating and
> popularity quality.
>
> Current intent deliberately dominates historical profile data. Hard
> requirements receive more influence than soft preferences. Known explicit
> exclusions are placed behind compliant products, while missing metadata is
> treated as unknown rather than as a mismatch.
>
> Budget handling follows the same principle: verified in-budget products lead,
> missing-price products remain eligible with uncertainty, and over-budget
> products are fallback options when too few viable alternatives exist.

### 5:30–5:55 — Recommendation and question policy

**Show:** Return to `starter/agent.py::_select_recommendations`, then the final
response dictionary in `respond`.

**Say:**

> After ranking, the agent avoids repeating products already shown under the
> same intent. Early turns use a focused four-item shortlist; later turns can
> expand to ten. Popularity can reorder only that already selected batch and is
> capped—it cannot change batch membership.
>
> If a customer declines an attribute, the agent can inspect candidate
> evidence and retarget the next question toward a color, material, or feature
> that actually separates the remaining products.

### 5:55–6:35 — Evaluator and evidence

**Show:** `evaluator/local_evaluator.py::evaluate`, then the public A/B table in
`SUBMISSION_REPORT.md`.

**Say:**

> The evaluator accepts only unique catalog-valid ASINs and checks exact
> equality. Intent-override sessions cannot succeed before the new intent is
> sent. Hit Rate measures whether we found the product, MRR rewards its rank,
> and MTTC rewards finding it earlier.
>
> Our lexical configuration scored 1.0 Hit Rate, 0.861851 MRR, 2.765 MTTC, and
> 0.923255 TechnicalScore on the public 200 sessions. With embeddings enabled
> at the conservative 0.75 calibration, TechnicalScore was 0.923318. It changed
> one public session, improving the target from rank eight to rank six without
> changing the hit turn.
>
> Increasing the semantic scale to 1.0 reduced TechnicalScore to 0.922945. We
> disclose that result because it demonstrates why semantic evidence must
> complement, rather than overpower, exact constraints.

### 6:35–7:00 — Limitations and close

**Show:** `SUBMISSION_REPORT.md`, “Limitations.”

**Say:**

> The parser remains primarily English, product metadata—especially price—is
> incomplete, and the first embedding-cache build must be prepared before an
> offline run. Our public semantic improvement is real but marginal, and public
> results are not a guarantee for the hidden sessions.
>
> The architecture is therefore deliberately resilient: structured dialog
> state keeps customer requirements in control, BM25 preserves exact lexical
> precision, MiniLM expands semantic recall, and deterministic ranking keeps
> every recommendation reproducible and inspectable.

## Public A/B results to display

| Configuration | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lexical/strict baseline | 1.000000 | 0.861851 | 2.765000 | 0.823500 | 0.923255 |
| Hybrid, semantic scale 0.75 | 1.000000 | 0.862060 | 2.765000 | 0.823500 | **0.923318** |
| Hybrid, semantic scale 1.00 | 1.000000 | 0.861149 | 2.770000 | 0.823000 | 0.922945 |

State clearly that these are public-development measurements, not hidden-set
results. The 0.75 setting changed only one of 200 sessions, improving its rank
from 8 to 6 at the same turn.

## Claims to avoid

Do not say:

- “The system is 75% semantic.” The `0.75` value is a score calibration
  multiplier, not a final percentage.
- “Embeddings replace BM25.” They add an optional parallel candidate route.
- “Embeddings rerank BM25 results.” Both routes generate candidates; the
  deterministic ranker reranks their union.
- “We trained MiniLM.” The system uses a legally accessible pretrained model.
- “There is no model.” Say there is no generative API; MiniLM is a local model.
- “The embedding improvement guarantees hidden performance.” It does not.
- “Token usage measures embedding computation.” The reported token fields are
  API prompt/completion tokens, so they remain zero.

## Likely judge questions

### Why not use only embeddings?

Exact product requirements—brand, color, material, size, and budget—are often
better represented lexically. The scale-1.0 regression empirically confirmed
that dense evidence can displace stronger exact matches when over-weighted.

### Why not use a vector database?

The frozen corpus is only 50,000 products. A `50,000 x 384` float32 matrix is
about 73 MiB, so exact NumPy similarity is fast, deterministic, and simpler to
reproduce than infrastructure-heavy approximate search.

### Is the semantic model required?

No. It is an optional local route. If dependencies or the model are missing,
the agent logs the failure and retains the complete lexical/strict path.

### When are embeddings computed?

Product vectors are computed once only when the catalog-specific cache is
missing. The query vector and its similarity against cached products are
computed for each new dialog-state signature. Identical repeated states use
the Agent's cache.

### Is `0.75` the semantic weight?

No. It transforms a semantic-only candidate's retrieval score as:

```text
retrieval_score = 0.75 * max(cosine_similarity, 0)
```

That retrieval score enters the ranker's 45% retrieval component. The semantic
route marker also contributes only inside the separate 5% route-evidence term.

