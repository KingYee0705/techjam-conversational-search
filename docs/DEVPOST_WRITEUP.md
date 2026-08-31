# Devpost Submission Draft

This document is written to be copied into Devpost. Replace every item marked
`[ADD ...]` before publishing, and remove this note from the final submission.

## Project name

**IntentWeave — Local, Explainable Conversational Product Search**

## One-line pitch

A local conversational shopping agent that combines structured intent memory,
BM25 search, and MiniLM embeddings to find the right product from 50,000
catalog items in ten turns or fewer.

## Short description

Customers change their minds and rarely know the perfect search keywords. Our
agent remembers evolving requirements, asks targeted follow-up questions, and
combines lexical and semantic retrieval with deterministic ranking—without a
generative API, billable tokens, or sending shopping intent to an external
service.

## Submitted configuration

The configuration presented in this write-up is the **hybrid agent with
semantic scale `0.75`**. Embeddings are activated through an explicit
environment flag so the result can be reproduced exactly. The same code keeps
the complete lexical/strict agent as a fallback, but running without the hybrid
variables reproduces the lexical score—not the `0.923318` headline score.

## Inspiration

Online shopping search usually assumes that the customer can express the
entire need in one query. Real conversations are messier. Someone may begin
with “I need shoes,” later specify trail running, reject leather, add a budget,
or say “actually, make them blue.” A useful shopping agent must understand
which preferences are still active, which ones were replaced, and what single
question would reduce uncertainty next.

The TechJam challenge made this problem concrete: find an exact hidden
`parent_asin` from a frozen catalog of 50,000 Clothing, Shoes, and Jewelry
products within at most ten turns. Success depends not only on recall, but also
on ranking the right product highly and finding it early.

We wanted to show that a practical shopping agent does not require an expensive
generative model on every turn. Careful conversational state, complementary
retrieval routes, and an interpretable ranker can already create a fast and
private experience. We then added a small local embedding model as a semantic
recall route for cases where the customer's words differ from the catalog.

## What it does

Our agent receives an anonymized preference profile and one customer message at
a time. On every turn it can ask one clarification question, return a ranked
list of catalog products, or do both.

The system:

- remembers active requirements across turns;
- distinguishes hard requirements from softer preferences;
- understands explicit negatives such as “avoid leather”;
- recognizes no-preference answers without adding garbage constraints;
- detects intent overrides and removes superseded preferences;
- asks a question from the competition's allowed attribute set;
- searches broad BM25, strict all-concept, and optional semantic routes;
- combines and deduplicates candidates from those routes;
- reranks products using the consolidated current-intent query and active constraints before
  historical profile evidence;
- avoids repeating the same products within an unchanged intent; and
- returns valid, unique `parent_asin` values through the required Agent API.

If the local embedding model is missing or semantic retrieval fails, the agent
continues with its full lexical and strict-search pipeline.

## Why it matters

Conversational product search can reduce the work of repeatedly changing
filters and rewriting queries, especially when the customer does not know the
catalog's vocabulary. Our design also addresses practical deployment concerns:

- **Privacy:** shopping intent stays on the evaluation machine after the local
  model is provisioned.
- **Cost:** there are no external inference calls, API credentials, or billable
  prompt/completion tokens.
- **Reliability:** deterministic outputs and a complete lexical fallback avoid
  dependence on one model or network service.
- **Auditability:** each result can be traced to dialog state, retrieval routes,
  explicit scoring signals, and a final ASIN tie-breaker.
- **Portability:** the system uses Python, SQLite, a small local model, and a
  reusable 74 MB vector cache instead of an infrastructure-heavy vector
  database.

These properties make the approach relevant beyond the competition: the same
architecture could support catalog assistants for retailers that need low
operating cost, predictable behavior, and control over customer data.

## What makes it innovative

Our innovation is not simply “adding embeddings.” It is the way conversation,
retrieval, and ranking remain separate but cooperate:

1. **State-aware intent replacement:** superseded values are removed from the
   active intent without being incorrectly treated as dislikes.
2. **Recall-first hybrid union:** broad BM25, strict all-concept, and MiniLM
   routes contribute candidates independently, so one route cannot silently
   erase another route's evidence.
3. **Conservative semantic calibration:** cosine similarity expands vocabulary
   coverage while exact current-message and constraint signals retain control.
4. **Unknown-aware ranking:** missing metadata remains eligible instead of
   becoming a false mismatch.
5. **Inspectable conversation policy:** clarification, shortlist size,
   exclusions, popularity movement, and failure fallbacks are bounded and
   deterministic.

## How we built it

### 1. Conversational intent memory

`starter/dialog.py` maintains isolated state for every session. Rather than
concatenating the entire conversation into an increasingly noisy query, it
stores active constraints, explicit negatives, replaced values, declined
attributes, the pending question, and turn history separately.

The parser uses the context of the last `ask_attribute` value to understand
short answers. It also gives an explicit override precedence over earlier
preferences. The result is a compact, retrieval-ready representation of the
customer's current intent.

### 2. Multi-route lexical retrieval

`starter/retrieval.py` loads the catalog into an in-memory SQLite FTS5 index.
We apply BM25 field weighting so title and category matches carry more evidence
than long descriptions.

We search several routes independently:

- the consolidated current intent;
- active constraints;
- the inferred product category;
- low-weight profile tags; and
- a strict route requiring all disclosed searchable concepts.

The broad routes protect recall, while the strict route rewards products that
satisfy the complete request. We union the routes so an empty strict result can
never delete good broad-search candidates.

### 3. Local embedding retrieval

`starter/embedding_retrieval.py` optionally uses
`sentence-transformers/all-MiniLM-L6-v2`. It converts bounded product documents
into normalized 384-dimensional vectors.

Product vectors are computed once when no compatible cache exists. The cache
includes a fingerprint of the frozen catalog, preventing vectors from one
catalog version from being silently reused with another. For each new dialog
intent, only one query vector is computed. Cosine similarity is then an exact
matrix multiplication against the cached 50,000-product matrix.

This is a candidate-union hybrid, not an “80% BM25 + 20% embeddings” shortcut.
Positive semantic similarity is first calibrated:

```text
semantic retrieval_score = 0.75 × max(cosine_similarity, 0)
```

Semantic candidates are then unioned with lexical candidates. When routes find
the same ASIN, we keep its strongest retrieval score and merge its route
provenance.

### 4. Deterministic constraint-aware ranking

`starter/ranking.py` applies six interpretable base signals:

- 45% retrieval evidence;
- 25% consolidated current-intent agreement;
- 20% active-constraint agreement;
- 5% route evidence;
- 3% profile overlap; and
- 2% product quality.

The newest intent is deliberately stronger than profile history. Rather than
treating missing attributes as failure, we score only known evidence. Confirmed
budget or negative-preference violations trail verified viable candidates,
while products with unknown values remain eligible.

### 5. Agent orchestration and response policy

`starter/agent.py` connects the conversation manager, lexical retriever,
embedding retriever, and ranker behind the required `reset()` and `respond()`
interface. It caches unchanged candidate states, selects unseen products, and
formats the exact evaluator response.

The agent uses deterministic customer-facing question templates. There is no
generative LLM call, no external inference service, and no billable prompt or
completion token usage during evaluation.

## Architecture

```text
User turn + aggregate profile
             |
             v
       Dialog state manager
             |
    +--------+---------+
    |        |         |
    v        v         v
 Broad    Strict    MiniLM
 BM25      AND      semantic
    |        |         |
    +--------+---------+
             |
             v
    Candidate union/deduplication
             |
             v
    Constraint-aware ranker
             |
             v
 Question + ranked recommendations
```

## Challenges we ran into

### Intent is state, not just text

Our first versions treated the conversation like one growing search query. That
failed when a customer said “actually” or “I don't mind.” Old values continued
to influence retrieval, and no-preference phrases could become accidental
constraints. Separating active, negative, superseded, and declined values made
the behavior much easier to reason about and test.

### Incomplete catalog metadata

Many products do not include every attribute, and price is frequently missing.
A strict filter would incorrectly eliminate plausible targets. We designed the
ranker to distinguish a confirmed mismatch from unknown evidence and used hard
filtering only when enough viable alternatives remain.

### Optimizing three competing metrics

Returning ten products immediately can reduce mean turns to conversion, but it
may place a weak match at a low rank. Smaller early batches allow later customer
answers to improve MRR, although they can delay the first hit. We measured
these policies separately and selected the disclosed tradeoff used by the
competition's TechnicalScore formula.

### Making embeddings useful without weakening exact requirements

Dense retrieval improved one public session at our conservative calibration,
but a larger scale caused four regressions and only two improvements. This was
a useful reminder that semantic similarity is complementary to lexical and
structured constraint evidence—it should not automatically dominate them.

### Reproducible local deployment

The first CPU embedding-cache build took several minutes. We added a reusable,
catalog-fingerprinted vector cache, local-only model loading, explicit
environment variables, and a complete BM25 fallback. This keeps the evaluated
configuration inspectable and avoids hidden network dependencies.

## Accomplishments that we are proud of

- Raised public TechnicalScore from the weak starter's `0.106710` to `0.923255`
  through dialog state, multi-route lexical retrieval, and constraint-aware ranking.
- The MiniLM route moved that score from `0.923255` to `0.923318` at scale
  `0.75`; we treat this as a marginal public-set result.
- Reached Hit Rate@10 `1.0` on all 200 released development sessions.
- Built explicit handling for Buying, Browsing, Boundary, and Intent Override
  behavior.
- Kept prompt and completion token usage at zero by using deterministic dialog
  logic and local retrieval.
- Added 151 deterministic unit and integration tests covering dialog,
  retrieval, embeddings, ranking, orchestration, and evaluator behavior.
- Preserved a complete lexical fallback when the optional model is unavailable.
- Made recommendations reproducible for a fixed catalog, model artifact,
  dependency environment, configuration, and cache.

## Public development results

| Configuration | Hit Rate@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: |
| Weak starter | 0.125000 | 0.068034 | 9.810 | 0.106710 |
| Lexical/strict agent | 1.000000 | 0.861851 | 2.765 | 0.923255 |
| **Hybrid, semantic scale 0.75** | **1.000000** | **0.862060** | **2.765** | **0.923318** |

These measurements use the released 200-session development set and the
unmodified local evaluator. They are not a guarantee of performance on the 800
hidden final sessions.

## Feasibility

On our development machine, the lexical agent initialized in approximately
1.61 seconds, averaged approximately 0.20 seconds per sampled response, and had
approximately 0.35-second sampled p95 latency. The embedding matrix contains
`50,000 × 384` float32 values and occupies approximately 74 MB. Building it
once on CPU took approximately 576.6 seconds; cached model and vector loading
took approximately 3.04 seconds. Query embedding plus exact similarity search
averaged approximately 0.016 seconds in our measurement.

The system uses no external inference API, no API credentials, and reports zero
API prompt/completion tokens. Estimated API/model usage cost is USD 0. Hardware and installation
costs are not included in that figure.

## What we learned

- Correct conversational memory can matter more than adding a larger model.
- Explicit “unknown” handling is essential when ranking real catalog data.
- Hybrid retrieval works best when lexical, strict, and semantic routes remain
  independently observable.
- Embedding calibration must be measured; more semantic weight is not
  automatically better.
- Public-set success should be reported honestly and separated from hidden-set
  expectations.
- Deterministic systems are easier to debug, reproduce, and explain to judges.

## What's next

With more time, we would:

- learn or validate different retrieval weights for Buying and Browsing
  sessions without inspecting private labels;
- improve typo tolerance, multilingual input, and unusual English paraphrases;
- extend candidate-aware question selection to more catalog attributes;
- evaluate a cross-encoder only on the final small candidate pool;
- add calibrated recommendation explanations showing which current constraints
  each product satisfies;
- use approximate nearest-neighbour search if the catalog grows far beyond
  50,000 items; and
- package the model and precomputed cache with a one-command integrity check for
  the final evaluation environment.

## Built with

- Python 3.10+
- SQLite FTS5 and BM25
- Sentence Transformers
- `sentence-transformers/all-MiniLM-L6-v2`
- NumPy
- PyTorch
- Python `unittest`
- Git and GitHub
- Amazon Reviews 2023-derived challenge data

## Try it

Default lexical mode:

```bash
python3 -m unittest
python3 -m evaluator.local_evaluator
```

Optional hybrid mode after provisioning the local model:

```bash
TECHJAM_ENABLE_EMBEDDINGS=1 \
TECHJAM_EMBEDDING_MODEL="$PWD/models/all-MiniLM-L6-v2-1110a243" \
TECHJAM_EMBEDDING_CACHE="$PWD/data/.embedding_cache" \
TECHJAM_SEMANTIC_SCORE_SCALE=0.75 \
python3 -m evaluator.local_evaluator --output results-hybrid.json
```

Full setup instructions are in the repository `README.md`.

## Demo video

**Video:** [ADD DEVPOST OR YOUTUBE VIDEO URL]

Suggested video sequence:

1. Show one multi-turn interaction and the final recommendation.
2. Open `Agent.respond()` as the orchestration entry point.
3. Follow the request into dialog state.
4. Show broad/strict BM25 and optional MiniLM as sibling candidate routes.
5. Return to the agent for candidate union.
6. Show the deterministic ranker and its weights.
7. Finish with the evaluator and public results.

The complete timed narration is in `docs/DEMO_VIDEO_WALKTHROUGH.md`.

## Team

- **[ADD NAME — Person 1]:** catalog indexing, lexical retrieval, and semantic
  candidate prototyping.
- **[ADD NAME — Person 2]:** constraint-aware ranking and scoring.
- **[ADD NAME — Person 3]:** conversation state, clarification, and intent
  overrides.
- **[ADD NAME — Person 4]:** Agent API orchestration, caching, and integration.
- **[ADD NAME — Person 5]:** evaluation, reproducibility, Git workflow, and
  submission documentation.

## Links

- **Source code:** [ADD FROZEN GITHUB COMMIT OR REPOSITORY URL]
- **Demo:** [ADD VIDEO URL]
- **Technical report:** [ADD DIRECT SUBMISSION_REPORT.MD URL]

## Final pre-publish checklist

- [ ] Choose the final project name.
- [ ] Replace all five team-member placeholders.
- [ ] Add the video URL and frozen repository/commit URL.
- [ ] Confirm the public metrics against the retained result artifacts.
- [ ] Confirm that the frozen final configuration uses hybrid scale `0.75` and
      the documented local model/cache paths.
- [ ] Provision and integrity-check the MiniLM model/cache if hybrid mode is
      submitted.
- [ ] Run `python3 -m unittest` and the unmodified final evaluator.
- [ ] Save the final `results.json`, commit SHA, environment details, and logs.
- [ ] Remove this checklist and the introductory editor note before pasting the
      write-up into Devpost.
