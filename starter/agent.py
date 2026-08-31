"""Competition entry point and end-to-end shopping-agent orchestrator.

Judge-facing data flow for one turn::

    Agent.respond
      -> DialogStateManager.process_turn       (understand current intent)
      -> CatalogRetriever + EmbeddingRetriever (generate candidate union)
      -> rank_products                         (score relevance/constraints)
      -> recommendation policy                 (unseen shortlist + API output)

The optional embedding route is local and has a deterministic BM25 fallback;
the public Agent contract remains identical in either mode.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import re
from collections import OrderedDict
from pathlib import Path

from starter.dialog import DialogStateManager
from starter.embedding_retrieval import (
    DEFAULT_MODEL as DEFAULT_EMBEDDING_MODEL,
    EmbeddingRetriever,
    semantic_query,
)
from starter.ranking import rank_products
from starter.retrieval import CatalogRetriever


LOGGER = logging.getLogger(__name__)
MAX_RECOMMENDATIONS = 10
EARLY_RECOMMENDATION_LIMIT = 4
EARLY_RECOMMENDATION_TURNS = 3
DEFAULT_CANDIDATE_POOL_SIZE = 200
DEFAULT_CANDIDATE_CACHE_SIZE = 32
MAX_POPULARITY_PROMOTION = 4.0
MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = Path("data/catalog.jsonl")
CATALOG_PATH_ENV = "TECHJAM_CATALOG_PATH"
ENABLE_EMBEDDINGS_ENV = "TECHJAM_ENABLE_EMBEDDINGS"
EMBEDDING_MODEL_ENV = "TECHJAM_EMBEDDING_MODEL"
EMBEDDING_CACHE_ENV = "TECHJAM_EMBEDDING_CACHE"
SEMANTIC_SCORE_SCALE_ENV = "TECHJAM_SEMANTIC_SCORE_SCALE"
DEFAULT_SEMANTIC_SCORE_SCALE = 0.75
CANDIDATE_QUESTION_LIMIT = 40
CANDIDATE_QUESTION_MIN_CANDIDATES = 8
CANDIDATE_QUESTION_MIN_COVERAGE = 0.35
CANDIDATE_QUESTION_MIN_SCORE = 0.50
CANDIDATE_QUESTION_HYSTERESIS = 0.10
CANDIDATE_ATTRIBUTE_PATTERNS = {
    "color": re.compile(
        r"\b(black|white|blue|navy|red|pink|green|brown|gray|grey|purple|"
        r"yellow|orange|beige|gold|silver|teal|maroon|khaki|tan|cream|"
        r"burgundy)\b",
        re.IGNORECASE,
    ),
    "material": re.compile(
        r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|linen|"
        r"cashmere|suede|velvet|rubber|acrylic|denim|fleece|canvas|satin|"
        r"mesh)\b",
        re.IGNORECASE,
    ),
    "feature": re.compile(
        r"\b(waterproof|water[- ]resistant|machine washable|hand wash|"
        r"zipper|zippered|button|pull[- ]on|pockets?|lightweight|breathable|"
        r"stretch|insulated|non[- ]slip|rubber sole|imported|buckle closure)\b",
        re.IGNORECASE,
    ),
}


def _resolve_catalog_path(catalog_path: str | Path | None) -> Path:
    """Resolve the catalog independently of the harness working directory."""

    if catalog_path is None:
        configured = os.environ.get(CATALOG_PATH_ENV)
        if not configured:
            return (MODULE_ROOT / DEFAULT_CATALOG_PATH).resolve()
    else:
        configured = catalog_path
    path = Path(configured).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (MODULE_ROOT / path).resolve()


def _environment_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _semantic_score_scale(value: object = None) -> float:
    configured = os.environ.get(SEMANTIC_SCORE_SCALE_ENV) if value is None else value
    if configured in (None, ""):
        return DEFAULT_SEMANTIC_SCORE_SCALE
    try:
        score = float(configured)
    except (TypeError, ValueError):
        return DEFAULT_SEMANTIC_SCORE_SCALE
    return max(0.0, min(1.0, score)) if math.isfinite(score) else DEFAULT_SEMANTIC_SCORE_SCALE


def _recommendation_limit(top_k: object, turn: object = None) -> int:
    if isinstance(top_k, bool):
        return 0
    try:
        value = int(top_k)
    except (TypeError, ValueError):
        return 0
    try:
        normalized_turn = int(turn)
    except (TypeError, ValueError):
        normalized_turn = EARLY_RECOMMENDATION_TURNS + 1
    maximum = (
        EARLY_RECOMMENDATION_LIMIT
        if 1 <= normalized_turn <= EARLY_RECOMMENDATION_TURNS
        else MAX_RECOMMENDATIONS
    )
    return max(0, min(maximum, value))


def _retrieval_score(candidate: object) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    try:
        score = float(candidate.get("retrieval_score", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def _rating_count(candidate: object) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    product = candidate.get("product")
    if not isinstance(product, dict):
        return 0.0
    value = product.get("rating_number", 0)
    if isinstance(value, bool):
        return 0.0
    try:
        count = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, count) if math.isfinite(count) else 0.0


def _flatten_product_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(
            f"{key} {_flatten_product_text(item)}" for key, item in value.items()
        )
    if isinstance(value, set):
        return " ".join(
            _flatten_product_text(item) for item in sorted(value, key=str)
        )
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten_product_text(item) for item in value)
    return str(value)


def _candidate_question_attribute(
    candidates: object,
    unavailable_attributes: set[str] | None = None,
    seen_asins: set[str] | None = None,
    current_attribute: str | None = None,
) -> str | None:
    """Choose a high-information field from the current candidate pool.

    This is intentionally conservative: color, material, and a controlled
    feature vocabulary have reasonably explicit catalog evidence.
    Sparse or one-valued evidence falls back to the dialog manager's normal
    question order rather than pretending a field will separate products.
    """

    if not isinstance(candidates, list):
        return None
    unavailable = unavailable_attributes or set()
    products: list[str] = []
    seen = seen_asins or set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        parent_asin = str(candidate.get("parent_asin", "")).strip()
        if not parent_asin or parent_asin in seen:
            continue
        product = candidate.get("product")
        if not isinstance(product, dict):
            product = candidate
        text = _flatten_product_text(product).strip()[:6000]
        if text:
            products.append(text)
            if len(products) >= CANDIDATE_QUESTION_LIMIT:
                break
    if len(products) < CANDIDATE_QUESTION_MIN_CANDIDATES:
        return None

    scored: list[tuple[float, str]] = []
    for attribute, pattern in CANDIDATE_ATTRIBUTE_PATTERNS.items():
        if attribute in unavailable:
            continue
        candidate_values = [
            {
                re.sub(r"[-\s]+", " ", match.group(1).lower())
                .replace("grey", "gray")
                .replace("zippered", "zipper")
                for match in pattern.finditer(text)
            }
            for text in products
        ]
        values = sorted(set().union(*candidate_values))
        if len(values) < 2:
            continue
        weights = [1.0 / math.log2(index + 2) for index in range(len(products))]
        total_weight = sum(weights)
        coverage = sum(
            weight for weight, found in zip(weights, candidate_values) if found
        ) / total_weight
        if coverage < CANDIDATE_QUESTION_MIN_COVERAGE:
            continue
        probabilities = [
            sum(
                weight
                for weight, found in zip(weights, candidate_values)
                if value in found
            ) / total_weight
            for value in values
        ]
        balanced = [probability for probability in probabilities if 0.10 <= probability <= 0.90]
        if not balanced:
            continue
        entropies = sorted(
            (
                -probability * math.log2(probability)
                - (1.0 - probability) * math.log2(1.0 - probability)
                for probability in probabilities
                if 0.0 < probability < 1.0
            ),
            reverse=True,
        )
        entropy = sum(entropies[:3]) / min(3, len(entropies))
        score = 0.40 * coverage + 0.60 * entropy
        if score >= CANDIDATE_QUESTION_MIN_SCORE:
            scored.append((score, attribute))
    if not scored:
        return None
    # Deterministic tie-break: color tends to be more directly answerable than
    # a blended fabric composition, then lexical order for future attributes.
    scored.sort(key=lambda item: (-item[0], item[1] != "color", item[1]))
    best_score, best_attribute = scored[0]
    current_score = next(
        (score for score, attribute in scored if attribute == current_attribute),
        None,
    )
    if best_attribute == current_attribute:
        return None
    if (
        current_score is not None
        and best_score < current_score + CANDIDATE_QUESTION_HYSTERESIS
    ):
        return None
    return best_attribute


def _fallback_rank(candidates: object) -> list[dict]:
    if not isinstance(candidates, list):
        return []
    valid = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict) and str(candidate.get("parent_asin", "")).strip()
    ]
    return sorted(
        valid,
        key=lambda candidate: (
            -_retrieval_score(candidate),
            str(candidate["parent_asin"]).strip(),
        ),
    )


def _candidate_signature(decision: dict, user_profile: dict, query: str) -> tuple:
    constraints = decision.get("active_constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}
    normalized_constraints = tuple(
        (
            str(attribute),
            tuple(str(value) for value in (values if isinstance(values, list) else [values])),
        )
        for attribute, values in sorted(constraints.items(), key=lambda item: str(item[0]))
    )
    tags = user_profile.get("preference_tags", []) if isinstance(user_profile, dict) else []
    if not isinstance(tags, list):
        tags = []
    return (
        query,
        str(decision.get("category") or ""),
        normalized_constraints,
        tuple(str(tag) for tag in tags),
    )


class Agent:
    """Public Agent API that coordinates state, retrieval, and ranking.

    No generative model or external API is called, so reported token usage and
    API cost remain zero. Dense retrieval, when enabled, uses a local model and
    falls back to the lexical path if its dependency or cache is unavailable.
    """

    def __init__(
        self,
        catalog_path: str | Path | None = None,
        *,
        candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
        candidate_cache_size: int = DEFAULT_CANDIDATE_CACHE_SIZE,
        embedding_retriever: object = None,
        enable_embeddings: bool | None = None,
        semantic_score_scale: float | None = None,
    ) -> None:
        self.catalog_path = _resolve_catalog_path(catalog_path)
        if not self.catalog_path.is_file():
            raise FileNotFoundError(
                f"Catalog not found at {self.catalog_path}. Set {CATALOG_PATH_ENV} "
                "or pass catalog_path explicitly."
            )
        self.candidate_pool_size = max(50, int(candidate_pool_size))
        self.candidate_cache_size = max(0, int(candidate_cache_size))

        # Startup stage A: the always-available lexical index over the frozen catalog.
        self.retriever = CatalogRetriever(self.catalog_path)

        # Startup stage B: optional local semantic index. It is deliberately
        # opt-in so the standard-library BM25 submission stays reproducible.
        self.embedding_retriever = embedding_retriever
        self.semantic_score_scale = _semantic_score_scale(semantic_score_scale)
        embeddings_enabled = (
            _environment_flag(ENABLE_EMBEDDINGS_ENV)
            if enable_embeddings is None
            else bool(enable_embeddings)
        )
        if self.embedding_retriever is None and embeddings_enabled:
            try:
                self.embedding_retriever = EmbeddingRetriever(
                    self.catalog_path,
                    model_name_or_path=os.environ.get(
                        EMBEDDING_MODEL_ENV, DEFAULT_EMBEDDING_MODEL
                    ),
                    cache_dir=os.environ.get(EMBEDDING_CACHE_ENV) or None,
                )
            except Exception as error:
                LOGGER.warning(
                    "Semantic retrieval unavailable; continuing with BM25: %s", error
                )
        self.dialog = DialogStateManager()
        self._sessions: dict[str, dict] = {}
        self._candidate_cache: OrderedDict[tuple, list[dict]] = OrderedDict()

    def reset(self, session_id: str, user_profile: dict) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        profile = copy.deepcopy(user_profile) if isinstance(user_profile, dict) else {}
        self._sessions[session_id] = {
            "user_profile": profile,
            "seen_asins": set(),
            "last_turn": None,
            "last_user_message": None,
            "last_response": None,
        }
        self.dialog.reset(session_id, profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        session = self._sessions[session_id]
        message = str(user_message or "")

        # 1. Make repeated harness calls idempotent. The public evaluator calls
        # respond twice with the same turn to verify deterministic behavior.
        if (
            session["last_turn"] == turn
            and session["last_user_message"] == message
            and session["last_response"] is not None
        ):
            return copy.deepcopy(session["last_response"])

        # 2. Convert free text into structured, evolving intent: category,
        # active/negative constraints, priorities, and the next question.
        decision = self.dialog.process_turn(session_id, message, turn)
        if decision["is_override"]:
            # The evaluator ignores hits before a new intent is sent. Products
            # shown under the old intent must therefore become eligible again.
            session["seen_asins"].clear()

        # 3. Generate a hybrid candidate pool from lexical, strict-conjunction,
        # and (when enabled) semantic routes.
        # Early turns use short, disjoint batches while the dialog is still
        # collecting preferences; later turns expand to the full Top 10 for
        # recall. This is a deliberate sequential-exploration tradeoff.
        limit = _recommendation_limit(top_k, turn)
        query = str(decision["search_query"] or message).strip()
        candidates = self._retrieve_candidates(decision, query, session["user_profile"])

        # 4. Rerank the union with current intent and hard/soft constraints.
        try:
            ranked = rank_products(
                candidates,
                query,
                decision["active_constraints"],
                session["user_profile"],
                top_k=self.candidate_pool_size,
                # Superseded override values are historical, not necessarily
                # disliked.  Only explicit negative preferences may penalize a
                # product downstream.
                excluded_constraints=decision.get("negative_constraints", {}),
                constraint_priorities=decision.get(
                    "constraint_priorities",
                    decision.get("constraint_provenance"),
                ),
            )
        except Exception as error:
            LOGGER.warning("Ranking failed; using deterministic retrieval order: %s", error)
            ranked = _fallback_rank(candidates)

        # 5. After "no preference", use the ranked pool to ask a field that
        # actually separates viable products instead of blindly repeating order.
        if decision.get("declined_attribute") and decision.get("ask_attribute"):
            dialog_state = self.dialog.get_state(session_id)
            current_attribute = str(decision["ask_attribute"])
            previously_asked = set(dialog_state.get("asked_attributes", []))
            previously_asked.discard(current_attribute)
            unavailable = {
                *dialog_state.get("declined_attributes", []),
                *previously_asked,
                *decision.get("active_constraints", {}).keys(),
            }
            adaptive_attribute = _candidate_question_attribute(
                ranked,
                unavailable,
                session["seen_asins"],
                current_attribute,
            )
            if adaptive_attribute:
                decision["message"] = self.dialog.retarget_question(
                    session_id, adaptive_attribute
                )
                decision["ask_attribute"] = adaptive_attribute

        # 6. Apply the turn-aware shortlist policy and avoid repeating products
        # already shown under the same intent.
        recommendations = self._select_recommendations(
            ranked,
            session["seen_asins"],
            limit,
        )
        session["seen_asins"].update(
            recommendation["parent_asin"] for recommendation in recommendations
        )

        # 7. Return exactly the organizer's Agent API shape. Internal scores and
        # product text never leak into the scored recommendation payload.
        response = {
            "message": self._customer_message(decision, bool(recommendations)),
            "ask_attribute": decision["ask_attribute"],
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        session["last_turn"] = turn
        session["last_user_message"] = message
        session["last_response"] = copy.deepcopy(response)
        return copy.deepcopy(response)

    def _retrieve_candidates(
        self,
        decision: dict,
        query: str,
        user_profile: dict,
    ) -> list[dict]:
        """Create one union of all enabled retrieval routes for this intent."""

        signature = _candidate_signature(decision, user_profile, query)

        # Identical dialog states reuse candidates across sessions/turn retries.
        cached = self._candidate_cache.get(signature)
        if cached is not None:
            self._candidate_cache.move_to_end(signature)
            return cached

        # Route group 1: weighted OR-style FTS searches for message,
        # constraints, category, and low-weight profile evidence.
        try:
            candidates = self.retriever.retrieve_products(
                query,
                active_constraints=decision["active_constraints"],
                user_profile=user_profile,
                category=decision["category"],
                top_k=self.candidate_pool_size,
            )
        except Exception as error:
            LOGGER.warning("Broad catalog retrieval failed: %s", error)
            return []
        if not isinstance(candidates, list):
            return []

        # Route group 2: a high-precision FTS conjunction over disclosed facts.
        try:
            strict_candidates = self.retriever.retrieve_strict_products(
                category=decision["category"],
                active_constraints=decision["active_constraints"],
                top_k=min(200, self.candidate_pool_size),
            )
        except Exception as error:
            LOGGER.warning("Strict catalog retrieval failed: %s", error)
            strict_candidates = []
        if isinstance(strict_candidates, list):
            candidates = [*candidates, *strict_candidates]

        # Route group 3: semantic nearest neighbours. This expands recall for
        # paraphrases; it does not replace exact lexical constraint matching.
        if self.embedding_retriever is not None:
            dense_query = semantic_query(
                query,
                decision.get("category"),
                decision.get("active_constraints"),
            )
            try:
                semantic_hits = self.embedding_retriever.retrieve(
                    dense_query,
                    top_k=self.candidate_pool_size,
                )
                if not isinstance(semantic_hits, list):
                    semantic_hits = []
                products = self.retriever.products_by_asin(
                    hit.get("parent_asin")
                    for hit in semantic_hits
                    if isinstance(hit, dict)
                )
                semantic_candidates: list[dict] = []
                for hit in semantic_hits:
                    if not isinstance(hit, dict):
                        continue
                    parent_asin = str(hit.get("parent_asin", "")).strip()
                    product = products.get(parent_asin)
                    if not parent_asin or product is None:
                        continue
                    try:
                        similarity = float(hit.get("semantic_similarity", 0.0))
                    except (TypeError, ValueError):
                        similarity = 0.0
                    if not math.isfinite(similarity):
                        similarity = 0.0
                    semantic_candidates.append({
                        "parent_asin": parent_asin,
                        "product": product,
                        # Keep semantic evidence conservative in the first A/B
                        # experiment. Lexical and structured constraints retain
                        # primary control of the final rank.
                        "retrieval_score": (
                            self.semantic_score_scale * max(0.0, min(1.0, similarity))
                        ),
                        "semantic_similarity": max(-1.0, min(1.0, similarity)),
                        "route_hits": ["semantic"],
                    })
                candidates = [*candidates, *semantic_candidates]
            except Exception as error:
                LOGGER.warning(
                    "Semantic candidate retrieval failed; using BM25 candidates: %s",
                    error,
                )

        if self.candidate_cache_size:
            self._candidate_cache[signature] = candidates
            self._candidate_cache.move_to_end(signature)
            while len(self._candidate_cache) > self.candidate_cache_size:
                self._candidate_cache.popitem(last=False)
        return candidates

    @staticmethod
    def _select_recommendations(
        ranked: object,
        seen_asins: set[str],
        limit: int,
    ) -> list[dict]:
        if limit <= 0 or not isinstance(ranked, list):
            return []

        unseen: list[dict] = []
        response_seen: set[str] = set()
        for candidate in ranked:
            if not isinstance(candidate, dict):
                continue
            parent_asin = str(candidate.get("parent_asin", "")).strip()
            if not parent_asin or parent_asin in response_seen:
                continue
            response_seen.add(parent_asin)
            if parent_asin not in seen_asins:
                unseen.append(candidate)
                if len(unseen) >= limit:
                    break

        # Preserve the relevance ranker's exact recommendation set, then use
        # catalog popularity only to adjust positions inside that bounded set.
        # This cannot remove a Top-K hit or change which products become seen,
        # and the key construction prevents promotion beyond the explicit cap.
        maximum_log_popularity = max(
            (math.log1p(_rating_count(candidate)) for candidate in unseen),
            default=0.0,
        )
        positioned = [
            (
                index,
                candidate,
                (
                    math.log1p(_rating_count(candidate)) / maximum_log_popularity
                    if maximum_log_popularity
                    else 0.0
                ),
            )
            for index, candidate in enumerate(unseen)
        ]
        positioned.sort(
            key=lambda item: (
                item[0] - MAX_POPULARITY_PROMOTION * item[2],
                -item[2],
                item[0],
                str(item[1].get("parent_asin", "")).strip(),
            )
        )

        return [
            {"parent_asin": str(candidate["parent_asin"]).strip()}
            for _, candidate, _ in positioned
        ]

    @staticmethod
    def _customer_message(decision: dict, has_recommendations: bool) -> str:
        question = str(decision.get("message") or "").strip()
        if decision.get("ask_attribute") is None:
            return question or "Here are my best matches based on your preferences."
        if has_recommendations:
            return f"Here are some options. {question}".strip()
        return question or "Could you share one more preference so I can narrow the search?"

    def close(self) -> None:
        self._candidate_cache.clear()
        close_embedding = getattr(self.embedding_retriever, "close", None)
        if callable(close_embedding):
            close_embedding()
        self.retriever.close()

    def __enter__(self) -> Agent:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
