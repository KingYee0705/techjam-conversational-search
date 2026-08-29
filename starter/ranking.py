from __future__ import annotations

import math
import re
from collections.abc import Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

FIELD_WEIGHTS = {
    "title": 1.00,
    "categories": 0.90,
    "store": 0.75,
    "features": 0.65,
    "details": 0.55,
    "description": 0.40,
}
ROUTE_WEIGHTS = {
    "current_message": 1.00,
    "active_constraints": 0.80,
    "category": 0.60,
    "profile": 0.25,
}
SCORE_WEIGHTS = {
    "retrieval": 0.45,
    "message": 0.25,
    "constraints": 0.20,
    "routes": 0.05,
    "profile": 0.03,
    "quality": 0.02,
}

MAX_BUDGET_PATTERNS = (
    re.compile(
        r"\b(?:under|below|less than|up to|at most|no more than|"
        r"max(?:imum)?(?:\s+of)?)\s*(?:usd\s*)?\$?\s*(\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"<=\s*(?:usd\s*)?\$?\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(
        r"(?:usd\s*)?\$\s*(\d+(?:\.\d+)?)\s*(?:or less|maximum|max)\b",
        re.IGNORECASE,
    ),
)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {_text(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_text(item) for item in value)
    return str(value)


def _tokens(value: object) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(_text(value))
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _normalized_phrase(value: object) -> str:
    return " ".join(token.lower() for token in TOKEN_RE.findall(_text(value)))


def _product(candidate: dict) -> dict:
    product = candidate.get("product")
    if isinstance(product, dict):
        return product
    # Gracefully support a flat product dictionary while the retriever is evolving.
    return candidate


def _field_corpora(product: dict) -> dict[str, tuple[set[str], str]]:
    result: dict[str, tuple[set[str], str]] = {}
    for field in FIELD_WEIGHTS:
        value = product.get(field)
        result[field] = (set(_tokens(value)), _normalized_phrase(value))
    return result


def _weighted_match(query: object, fields: dict[str, tuple[set[str], str]]) -> float:
    query_tokens = list(dict.fromkeys(_tokens(query)))
    if not query_tokens:
        return 0.0

    token_total = 0.0
    for token in query_tokens:
        token_total += max(
            (
                weight
                for field, weight in FIELD_WEIGHTS.items()
                if token in fields[field][0]
            ),
            default=0.0,
        )
    token_score = token_total / len(query_tokens)

    phrases = {
        " ".join(query_tokens[index:index + size])
        for size in (2, 3)
        for index in range(len(query_tokens) - size + 1)
    }
    if not phrases:
        return min(1.0, token_score)

    phrase_total = 0.0
    for phrase in phrases:
        phrase_total += max(
            (
                weight
                for field, weight in FIELD_WEIGHTS.items()
                if phrase in fields[field][1]
            ),
            default=0.0,
        )
    phrase_score = phrase_total / len(phrases)
    return min(1.0, 0.85 * token_score + 0.15 * phrase_score)


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _unit_score(value: object) -> float:
    return max(0.0, min(1.0, _number(value)))


def _price(product: dict) -> float | None:
    value = product.get("price")
    if value is None or isinstance(value, bool):
        return None
    number = _number(value, default=-1.0)
    return number if number >= 0.0 else None


def _constraint_items(active_constraints: object) -> list[str]:
    if not isinstance(active_constraints, dict):
        return []
    values: list[str] = []
    for attribute, items in active_constraints.items():
        if str(attribute).lower() == "budget":
            continue
        if isinstance(items, (str, int, float)):
            items = [items]
        if not isinstance(items, Iterable) or isinstance(items, dict):
            continue
        values.extend(_text(item).strip() for item in items if _text(item).strip())
    return values


def _budget_text(active_constraints: object, user_message: str) -> str:
    values: list[str] = [user_message]
    if isinstance(active_constraints, dict):
        budget = active_constraints.get("budget", [])
        if isinstance(budget, (str, int, float)):
            budget = [budget]
        if isinstance(budget, Iterable) and not isinstance(budget, dict):
            values.extend(_text(item) for item in budget)
    return " ".join(values)


def _maximum_budget(active_constraints: object, user_message: str) -> float | None:
    text = _budget_text(active_constraints, user_message)
    matches: list[float] = []
    for pattern in MAX_BUDGET_PATTERNS:
        for match in pattern.finditer(text):
            value = _number(match.group(1), default=-1.0)
            if value >= 0.0:
                matches.append(value)
    return min(matches) if matches else None


def _constraint_score(
    active_constraints: object,
    fields: dict[str, tuple[set[str], str]],
) -> float:
    items = _constraint_items(active_constraints)
    if not items:
        return 0.0
    return sum(_weighted_match(item, fields) for item in items) / len(items)


def _profile_score(user_profile: object, fields: dict[str, tuple[set[str], str]]) -> float:
    if not isinstance(user_profile, dict):
        return 0.0
    tags = user_profile.get("preference_tags", [])
    if not isinstance(tags, list):
        return 0.0
    return _weighted_match(tags, fields)


def _route_score(candidate: dict) -> float:
    routes = candidate.get("route_hits", [])
    if not isinstance(routes, (list, tuple, set)):
        return 0.0
    unique_routes = {str(route).strip().lower() for route in routes}
    return min(1.0, sum(ROUTE_WEIGHTS.get(route, 0.0) for route in unique_routes))


def _quality_score(product: dict, max_log_ratings: float) -> float:
    rating = max(0.0, min(5.0, _number(product.get("average_rating")))) / 5.0
    rating_count = max(0.0, _number(product.get("rating_number")))
    popularity = math.log1p(rating_count) / max_log_ratings if max_log_ratings else 0.0
    return 0.70 * rating + 0.30 * popularity


def _copy_candidate(candidate: dict, parent_asin: str) -> dict:
    copied = dict(candidate)
    copied["parent_asin"] = parent_asin
    product = candidate.get("product")
    if isinstance(product, dict):
        copied["product"] = dict(product)
    routes = candidate.get("route_hits")
    if isinstance(routes, (list, tuple, set)):
        copied["route_hits"] = list(routes)
    return copied


def _deduplicate(candidates: object) -> list[dict]:
    if not isinstance(candidates, list):
        return []
    unique: dict[str, dict] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        parent_asin = str(candidate.get("parent_asin", "")).strip()
        if not parent_asin:
            continue
        copied = _copy_candidate(candidate, parent_asin)
        existing = unique.get(parent_asin)
        if existing is None:
            unique[parent_asin] = copied
            continue

        existing_routes = existing.get("route_hits", [])
        copied_routes = copied.get("route_hits", [])
        merged_routes = sorted({
            *[str(route) for route in existing_routes if str(route).strip()],
            *[str(route) for route in copied_routes if str(route).strip()],
        })
        better = copied if _unit_score(copied.get("retrieval_score")) > _unit_score(
            existing.get("retrieval_score")
        ) else existing
        merged = _copy_candidate(better, parent_asin)
        if merged_routes:
            merged["route_hits"] = merged_routes
        unique[parent_asin] = merged
    return list(unique.values())


def rank_products(
    candidates: list[dict],
    user_message: str,
    active_constraints: dict[str, list[str]],
    user_profile: dict,
    top_k: int = 10,
) -> list[dict]:
    """Return deterministic, ranked copies of retrieval candidates.

    Each candidate should contain ``parent_asin``, a nested ``product`` catalog
    record, a normalized ``retrieval_score`` (larger is better), and optional
    ``route_hits``. Incomplete candidates are tolerated and inputs are not mutated.
    """

    try:
        limit = max(0, int(top_k))
    except (TypeError, ValueError):
        return []
    if limit == 0:
        return []

    unique = _deduplicate(candidates)
    if not unique:
        return []

    max_log_ratings = max(
        (math.log1p(max(0.0, _number(_product(item).get("rating_number")))) for item in unique),
        default=0.0,
    )

    scored: list[tuple[float, str, dict, float | None]] = []
    for candidate in unique:
        parent_asin = str(candidate["parent_asin"])
        product = _product(candidate)
        fields = _field_corpora(product)
        score = (
            SCORE_WEIGHTS["retrieval"] * _unit_score(candidate.get("retrieval_score"))
            + SCORE_WEIGHTS["message"] * _weighted_match(user_message, fields)
            + SCORE_WEIGHTS["constraints"] * _constraint_score(active_constraints, fields)
            + SCORE_WEIGHTS["routes"] * _route_score(candidate)
            + SCORE_WEIGHTS["profile"] * _profile_score(user_profile, fields)
            + SCORE_WEIGHTS["quality"] * _quality_score(product, max_log_ratings)
        )
        scored.append((score, parent_asin, candidate, _price(product)))

    def relevance_key(item: tuple[float, str, dict, float | None]) -> tuple[float, str]:
        return (-item[0], item[1])

    maximum_budget = _maximum_budget(active_constraints, user_message)
    if maximum_budget is None:
        return [item[2] for item in sorted(scored, key=relevance_key)[:limit]]

    within_budget = [item for item in scored if item[3] is not None and item[3] <= maximum_budget]
    unknown_price = [item for item in scored if item[3] is None]
    over_budget = [item for item in scored if item[3] is not None and item[3] > maximum_budget]

    within_budget.sort(key=relevance_key)
    # Missing-price products remain eligible but follow verified in-budget products.
    unknown_price.sort(key=lambda item: (-(item[0] - 0.10), item[1]))
    # When fallback is necessary, prefer the smallest budget violation first.
    over_budget.sort(
        key=lambda item: (
            (item[3] - maximum_budget) / max(maximum_budget, 1.0),
            -item[0],
            item[1],
        )
    )

    viable = [*within_budget, *unknown_price]
    ordered = viable if len(viable) >= 10 else [*viable, *over_budget]
    return [item[2] for item in ordered[:limit]]
