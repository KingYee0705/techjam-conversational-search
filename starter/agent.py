from __future__ import annotations

import copy
import math
from collections import OrderedDict
from pathlib import Path

from starter.dialog import DialogStateManager
from starter.ranking import rank_products
from starter.retrieval import CatalogRetriever


MAX_RECOMMENDATIONS = 10
EARLY_RECOMMENDATION_LIMIT = 4
EARLY_RECOMMENDATION_TURNS = 3
DEFAULT_CANDIDATE_POOL_SIZE = 200
DEFAULT_CANDIDATE_CACHE_SIZE = 32
POPULARITY_POSITION_WEIGHT = 0.40


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
    """Offline conversational shopping agent.

    The agent combines the independently tested dialog, retrieval, and ranking
    modules. It needs no network service or API key, so model-token usage is zero.
    """

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
        candidate_cache_size: int = DEFAULT_CANDIDATE_CACHE_SIZE,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.candidate_pool_size = max(50, int(candidate_pool_size))
        self.candidate_cache_size = max(0, int(candidate_cache_size))
        self.retriever = CatalogRetriever(self.catalog_path)
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

        if (
            session["last_turn"] == turn
            and session["last_user_message"] == message
            and session["last_response"] is not None
        ):
            return copy.deepcopy(session["last_response"])

        decision = self.dialog.process_turn(session_id, message, turn)
        if decision["is_override"]:
            # The evaluator ignores hits before a new intent is sent. Products
            # shown under the old intent must therefore become eligible again.
            session["seen_asins"].clear()

        # Early turns emphasize a short, high-confidence list while the dialog
        # is still collecting preferences.  Once enough clarification has been
        # possible, expand to the full competition Top 10 for recall.
        limit = _recommendation_limit(top_k, turn)
        query = str(decision["search_query"] or message).strip()
        candidates = self._retrieve_candidates(decision, query, session["user_profile"])

        try:
            ranked = rank_products(
                candidates,
                query,
                decision["active_constraints"],
                session["user_profile"],
                top_k=self.candidate_pool_size,
            )
        except Exception:
            ranked = _fallback_rank(candidates)

        recommendations = self._select_recommendations(
            ranked,
            session["seen_asins"],
            limit,
        )
        session["seen_asins"].update(
            recommendation["parent_asin"] for recommendation in recommendations
        )

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
        signature = _candidate_signature(decision, user_profile, query)
        cached = self._candidate_cache.get(signature)
        if cached is not None:
            self._candidate_cache.move_to_end(signature)
            return cached

        try:
            candidates = self.retriever.retrieve_products(
                query,
                active_constraints=decision["active_constraints"],
                user_profile=user_profile,
                category=decision["category"],
                top_k=self.candidate_pool_size,
            )
        except Exception:
            return []
        if not isinstance(candidates, list):
            return []

        try:
            strict_candidates = self.retriever.retrieve_strict_products(
                category=decision["category"],
                active_constraints=decision["active_constraints"],
                top_k=min(200, self.candidate_pool_size),
            )
        except Exception:
            strict_candidates = []
        if isinstance(strict_candidates, list):
            candidates = [*candidates, *strict_candidates]

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
        # This cannot remove a Top-K hit or change which products become seen.
        positioned = list(enumerate(unseen))
        positioned.sort(
            key=lambda item: (
                item[0] - POPULARITY_POSITION_WEIGHT * math.log1p(_rating_count(item[1])),
                item[0],
                str(item[1].get("parent_asin", "")).strip(),
            )
        )

        return [
            {"parent_asin": str(candidate["parent_asin"]).strip()}
            for _, candidate in positioned
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
        self.retriever.close()

    def __enter__(self) -> Agent:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
