from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Iterable
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}
ROUTE_WEIGHTS = {
    "current_message": 1.00,
    "active_constraints": 0.85,
    "category": 0.65,
    "profile": 0.25,
}
ROUTE_ORDER = tuple(ROUTE_WEIGHTS)
STRICT_SCORE_FLOOR = 0.60

SYNONYM_GROUPS = (
    {"shoe", "shoes", "sneaker", "sneakers", "footwear"},
    {"shirt", "shirts", "tee", "tees", "tshirt"},
    {"pants", "trousers", "slacks"},
    {"purse", "purses", "handbag", "handbags"},
    {"jacket", "jackets", "coat", "coats"},
    {"earring", "earrings"},
    {"necklace", "necklaces", "pendant", "pendants"},
)
SYNONYMS = {
    term: tuple(sorted(group - {term}))
    for group in SYNONYM_GROUPS
    for term in group
}
STRICT_LABEL_RE = re.compile(
    r"^\s*(?:color|material|feature|style|size|brand|use[_ -]?case)\s*:\s*",
    re.IGNORECASE,
)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {_text(item)}" for key, item in value.items())
    if isinstance(value, set):
        return " ".join(_text(item) for item in sorted(value, key=str))
    if isinstance(value, (list, tuple)):
        return " ".join(_text(item) for item in value)
    return str(value)


def _base_terms(value: object) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(_text(value))
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _inflections(term: str) -> tuple[str, ...]:
    variants: set[str] = set()
    if len(term) > 3 and term.endswith("ies"):
        variants.add(term[:-3] + "y")
    elif len(term) > 4 and term.endswith(("sses", "shes", "ches", "xes", "zes")):
        variants.add(term[:-2])
    elif len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        variants.add(term[:-1])
    elif len(term) > 2 and term.endswith("y") and term[-2] not in "aeiou":
        variants.add(term[:-1] + "ies")
    elif term.endswith(("s", "x", "z", "ch", "sh")):
        variants.add(term + "es")
    elif len(term) > 2:
        variants.add(term + "s")
    return tuple(sorted(variants))


def _expanded_terms(value: object, limit: int = 60) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for term in _base_terms(value):
        for candidate in (term, *_inflections(term), *SYNONYMS.get(term, ())):
            if candidate not in seen:
                expanded.append(candidate)
                seen.add(candidate)
                if len(expanded) >= limit:
                    return expanded
    return expanded


def _fts_expression(value: object) -> str:
    return " OR ".join(f'"{term}"' for term in _expanded_terms(value))


def _strict_fts_expression(category: object, active_constraints: object) -> str:
    """Build a high-precision query while retaining inflection alternatives.

    Broad retrieval uses OR across every query token, which is useful when a
    customer's wording differs from incomplete catalog text.  Once category or
    constraint evidence is available, this companion expression requires every
    disclosed concept while still allowing synonyms and singular/plural forms.
    """

    values: list[str] = []
    if category is not None and str(category).strip():
        values.append(str(category).strip())
    if isinstance(active_constraints, dict):
        for attribute, items in active_constraints.items():
            if str(attribute).strip().lower() in {"category", "budget"}:
                continue
            if isinstance(items, (str, int, float)):
                items = [items]
            if not isinstance(items, Iterable) or isinstance(items, dict):
                continue
            for item in items:
                cleaned = STRICT_LABEL_RE.sub("", str(item)).strip()
                if cleaned:
                    values.append(cleaned)

    groups: list[str] = []
    for term in dict.fromkeys(_base_terms(values)):
        variants = dict.fromkeys((term, *_inflections(term), *SYNONYMS.get(term, ())))
        groups.append("(" + " OR ".join(f'\"{variant}\"' for variant in variants) + ")")
    return " AND ".join(groups)


def _constraint_query(active_constraints: object) -> str:
    if not isinstance(active_constraints, dict):
        return ""
    values: list[str] = []
    for attribute, items in active_constraints.items():
        # Price is structured data used by the ranker, not searchable FTS text.
        if str(attribute).strip().lower() == "budget":
            continue
        if isinstance(items, (str, int, float)):
            items = [items]
        if not isinstance(items, Iterable) or isinstance(items, dict):
            continue
        rendered = [str(item).strip() for item in items if str(item).strip()]
        if rendered:
            values.append(str(attribute))
            values.extend(rendered)
    return " ".join(values)


def _category_query(category: object, active_constraints: object) -> str:
    values: list[str] = []
    if category is not None and str(category).strip():
        values.append(str(category).strip())
    if isinstance(active_constraints, dict):
        categories = active_constraints.get("category", [])
        if isinstance(categories, (str, int, float)):
            categories = [categories]
        if isinstance(categories, Iterable) and not isinstance(categories, dict):
            values.extend(str(item).strip() for item in categories if str(item).strip())
    return " ".join(dict.fromkeys(values))


def _profile_query(user_profile: object) -> str:
    if not isinstance(user_profile, dict):
        return ""
    tags = user_profile.get("preference_tags", [])
    if not isinstance(tags, list):
        return ""
    return " ".join(str(tag).strip() for tag in tags if str(tag).strip())


def _route_scores(raw_scores: list[float]) -> list[float]:
    """Convert SQLite BM25 values into stable scores where larger is better."""

    if not raw_scores:
        return []
    strengths = [max(0.0, -score) if math.isfinite(score) else 0.0 for score in raw_scores]
    strongest = max(strengths)
    weakest = min(strengths)
    count = len(strengths)
    normalized: list[float] = []
    for index, strength in enumerate(strengths):
        if strongest == weakest:
            magnitude = 1.0
        else:
            magnitude = (strength - weakest) / (strongest - weakest)
        rank_component = 1.0 if count == 1 else 1.0 - index / (count - 1)
        normalized.append(max(0.0, min(1.0, 0.75 * magnitude + 0.25 * rank_component)))
    return normalized


class CatalogRetriever:
    """Offline multi-route FTS5 retriever for the frozen product catalog."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, product_json UNINDEXED, title, categories, features, "
            "details, store, description, tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        json.dumps(product, ensure_ascii=False, separators=(",", ":")),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def _search_route(self, query: object, limit: int) -> list[tuple[str, dict, float]]:
        expression = _fts_expression(query)
        if not expression or limit <= 0:
            return []
        return self._search_expression(expression, limit)

    def _search_expression(self, expression: str, limit: int) -> list[tuple[str, dict, float]]:
        if not expression or limit <= 0:
            return []
        rows = self.connection.execute(
            "SELECT parent_asin, product_json, "
            "bm25(products, 0.0, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS score "
            "FROM products WHERE products MATCH ? ORDER BY score, parent_asin LIMIT ?",
            (expression, limit),
        ).fetchall()
        raw_scores = [float(row[2]) for row in rows]
        normalized = _route_scores(raw_scores)
        return [
            (str(row[0]), json.loads(str(row[1])), score)
            for row, score in zip(rows, normalized)
        ]

    def retrieve_products(
        self,
        query: str,
        *,
        active_constraints: dict[str, list[str]] | None = None,
        user_profile: dict | None = None,
        category: str | None = None,
        top_k: int = 200,
    ) -> list[dict]:
        """Return ranked, deduplicated candidates for the downstream ranker.

        ``retrieval_score`` is normalized to [0, 1], with larger values better.
        ``route_hits`` reports which independent queries found each product.
        """

        try:
            limit = max(0, int(top_k))
        except (TypeError, ValueError):
            return []
        if limit == 0:
            return []

        constraints = active_constraints or {}
        profile = user_profile or {}
        route_queries = {
            "current_message": query,
            "active_constraints": _constraint_query(constraints),
            "category": _category_query(category, constraints),
            "profile": _profile_query(profile),
        }
        active_routes = [route for route in ROUTE_ORDER if _fts_expression(route_queries[route])]
        if not active_routes:
            return []

        route_limit = min(500, max(50, limit * 2))
        total_weight = sum(ROUTE_WEIGHTS[route] for route in active_routes)
        merged: dict[str, dict] = {}
        score_parts: dict[str, float] = {}

        for route in active_routes:
            for parent_asin, product, score in self._search_route(route_queries[route], route_limit):
                if parent_asin not in merged:
                    merged[parent_asin] = {
                        "parent_asin": parent_asin,
                        "product": product,
                        "retrieval_score": 0.0,
                        "route_hits": [],
                    }
                    score_parts[parent_asin] = 0.0
                merged[parent_asin]["route_hits"].append(route)
                score_parts[parent_asin] += ROUTE_WEIGHTS[route] * score

        combined_scores = {
            parent_asin: max(0.0, score_parts[parent_asin] / total_weight)
            for parent_asin in merged
        }
        strongest_combined = max(combined_scores.values(), default=0.0)
        for parent_asin, candidate in merged.items():
            candidate["retrieval_score"] = (
                min(1.0, combined_scores[parent_asin] / strongest_combined)
                if strongest_combined
                else 0.0
            )
            route_set = set(candidate["route_hits"])
            candidate["route_hits"] = [route for route in ROUTE_ORDER if route in route_set]

        ordered = sorted(
            merged.values(),
            key=lambda candidate: (-candidate["retrieval_score"], candidate["parent_asin"]),
        )
        return ordered[:limit]

    def retrieve_strict_products(
        self,
        *,
        category: str | None = None,
        active_constraints: dict[str, list[str]] | None = None,
        top_k: int = 200,
    ) -> list[dict]:
        """Return products satisfying every disclosed searchable concept.

        This bounded precision route is kept separate from broad retrieval so
        both methods respect their own ``top_k`` contract.  The agent unions
        the two pools before reranking; an empty over-constrained result cannot
        remove broad candidates.
        """

        try:
            limit = max(0, int(top_k))
        except (TypeError, ValueError):
            return []
        if limit == 0:
            return []

        strict_expression = _strict_fts_expression(category, active_constraints or {})
        if not strict_expression:
            return []
        try:
            strict_results = self._search_expression(strict_expression, limit)
        except sqlite3.OperationalError:
            return []
        return [
            {
                "parent_asin": parent_asin,
                "product": product,
                # A product satisfying every disclosed concept deserves enough
                # evidence to reach the downstream reranker even when it sits
                # deep among many generic OR matches.  The conservative floor
                # was chosen below the strongest route score and broad search
                # remains available alongside it.
                "retrieval_score": max(STRICT_SCORE_FLOOR, strict_score),
                "route_hits": ["current_message", "active_constraints"],
            }
            for parent_asin, product, strict_score in strict_results
        ]

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> CatalogRetriever:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
