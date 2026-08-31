"""Constraint-aware deterministic reranking for the merged candidate pool.

The ranker is intentionally source-agnostic: BM25, strict, and semantic
candidates share one contract. It combines retrieval evidence with direct
message/constraint coverage, then applies safe budget and exclusion ordering.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
SCALAR_BOUNDARY = "\u241e"
CLAUSE_BOUNDARY_RE = re.compile(r"(?:[;!?]+|(?<!\d)\.(?!\d)|\s+[—–]\s+)")
SIZE_VALUE_PATTERN = (
    r"(?<![a-z0-9])(?:extra\s+small|extra\s+large|small|medium|large|"
    r"xxxl|xxl|xxs|xl|xs|s|m|l|\d+(?:\.\d+)?)(?![a-z0-9])"
)
SIZE_VALUE_RE = re.compile(SIZE_VALUE_PATTERN, re.IGNORECASE)
# Keep prose-derived markers inside the immediate ``size`` clause. A wider
# trailing window can turn a later model height such as 5'8.5" into size 8.5.
SIZE_CONTEXT_RE = re.compile(
    r"\bsizes?\b(?:\s+charts?)?"
    r"(?:\s+(?:available|offered|include(?:s)?|are|is))?"
    r"\s*(?::|=|-)?\s*"
    rf"(?P<values>(?:(?:us|uk|eu)\s+)?{SIZE_VALUE_PATTERN}"
    rf"(?:\s*(?:[,/|&]|[-–]|\band\b|\bor\b)\s*"
    rf"(?:(?:us|uk|eu)\s+)?{SIZE_VALUE_PATTERN})*)",
    re.IGNORECASE,
)
# Numeric dimensions are measurements even when catalog prose labels them
# "size" (for example, "model size is 5'8.5\"").
SIZE_MEASUREMENT_RE = re.compile(
    r"^\s*\d+(?:\.\d+)?\s*(?:"
    r"['’′]\s*\d+(?:\.\d+)?\s*(?:[\"”″]|inches?)?|"
    r"[x×]\s*\d+(?:\.\d+)?|"
    r"(?:-\s*)?(?:[\"”″]|inches?\b|in\b|feet\b|foot\b|ft\b|"
    r"millimeters?\b|mm\b|centimeters?\b|cm\b|meters?\b))",
    re.IGNORECASE,
)
SIZE_DIMENSION_RE = re.compile(
    r"(?<![a-z0-9])\d+(?:\.\d+)?"
    r"(?:\s*[x×]\s*\d+(?:\.\d+)?){1,}"
    r"(?:\s*[-–]?\s*(?:[\"”″]|inches?\b|in\b|feet\b|foot\b|ft\b|"
    r"millimeters?\b|mm\b|centimeters?\b|cm\b|meters?\b))?",
    re.IGNORECASE,
)
SIZE_ALIASES = {
    "extra small": "xs",
    "small": "s",
    "medium": "m",
    "large": "l",
    "extra large": "xl",
}
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
    "semantic": 0.45,
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
# The weights sum to one and encode the design priority: the current request
# and active constraints dominate the small historical/profile and popularity
# priors. Semantic candidates enter through the normalized retrieval signal.
CONSTRAINT_PRIORITY_WEIGHTS = {
    "hard": 1.5,
    "normal": 1.0,
    "soft": 0.75,
}
HARD_CONSTRAINT_MATCH_BOOST = 0.30
HARD_PRIORITY_NAMES = {
    "hard", "required", "requirement", "must", "must_have", "non_negotiable",
    "initial_requirement", "override",
}
SOFT_PRIORITY_NAMES = {
    "soft", "optional", "preference", "preferred", "initial_preference",
}
CONSTRAINT_LABEL_RE = re.compile(
    r"^\s*(?:category|material|color|size|style|brand|budget|feature|"
    r"use[_ -]?case|other)\s*:\s*",
    re.IGNORECASE,
)
NEGATION_CONTEXT_TOKENS = {
    "avoid", "avoiding", "avoids", "never", "no", "non", "not", "without",
}

MONEY_NUMBER_PATTERN = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
MAX_BUDGET_PATTERNS = (
    re.compile(
        r"\b(?:under|below|less than|up to|at most|no more than|"
        rf"max(?:imum)?(?:\s+of)?)\s*(?:usd\s*)?\$?\s*({MONEY_NUMBER_PATTERN})\b",
        re.IGNORECASE,
    ),
    re.compile(rf"<=\s*(?:usd\s*)?\$?\s*({MONEY_NUMBER_PATTERN})\b", re.IGNORECASE),
    re.compile(
        rf"(?:usd\s*)?\$\s*({MONEY_NUMBER_PATTERN})\s*(?:or less|maximum|max)\b",
        re.IGNORECASE,
    ),
)
BUDGET_RANGE_PATTERNS = (
    re.compile(
        rf"\b(?:between|from)\s*(?:(?:usd\s*)?\$|usd\s+)\s*({MONEY_NUMBER_PATTERN})\s*"
        rf"(?:and|to|-)\s*(?:usd\s*)?\$?\s*({MONEY_NUMBER_PATTERN})\b",
        re.IGNORECASE,
    ),
    # A currency symbol makes a compact range unambiguously monetary even
    # without introductory words (for example ``$50-$100``).
    re.compile(
        rf"(?<![a-z0-9])(?:usd\s*)?\$\s*({MONEY_NUMBER_PATTERN})\s*"
        rf"(?:-|–|—|to)\s*(?:usd\s*)?\$?\s*({MONEY_NUMBER_PATTERN})\b",
        re.IGNORECASE,
    ),
    # An explicit budget/cost/spend label is also unambiguous without a
    # currency symbol: ``budget between 50 and 100``.
    re.compile(
        rf"\b(?:budget|price|cost|spend|spending)\b[^.;!?]{{0,24}}?"
        rf"\b(?:between|from)\s*({MONEY_NUMBER_PATTERN})\s*"
        rf"(?:and|to|-)\s*({MONEY_NUMBER_PATTERN})\b",
        re.IGNORECASE,
    ),
)
STRUCTURED_BUDGET_RANGE_PATTERNS = (
    re.compile(
        rf"\b(?:between|from)\s*(?:usd\s*)?\$?\s*({MONEY_NUMBER_PATTERN})\s*"
        rf"(?:and|to|-)\s*(?:usd\s*)?\$?\s*({MONEY_NUMBER_PATTERN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?<![a-z0-9])(?:usd\s*)?\$?\s*({MONEY_NUMBER_PATTERN})\s*"
        rf"(?:-|–|—|to)\s*(?:usd\s*)?\$?\s*({MONEY_NUMBER_PATTERN})(?![a-z0-9])",
        re.IGNORECASE,
    ),
)
BUDGET_TARGET_PATTERNS = (
    re.compile(
        r"\b(?:budget\s+)?(?:around|about|approximately|roughly|near)\s*"
        rf"(?:usd\s*)?\$?\s*({MONEY_NUMBER_PATTERN})\b",
        re.IGNORECASE,
    ),
)
MIN_BUDGET_PATTERNS = (
    re.compile(
        r"\b(?:at least|minimum(?:\s+of)?|over|more than)\s*"
        rf"(?:usd\s*)?\$?\s*({MONEY_NUMBER_PATTERN})\b",
        re.IGNORECASE,
    ),
    re.compile(rf">=\s*(?:usd\s*)?\$?\s*({MONEY_NUMBER_PATTERN})\b", re.IGNORECASE),
)
CONTEXTUAL_BUDGET_NUMBER_RE = re.compile(
    rf"(?<![a-z0-9]){MONEY_NUMBER_PATTERN}(?![a-z0-9])",
    re.IGNORECASE,
)
NON_BUDGET_SIZE_COMPARISON_RE = re.compile(
    rf"\bsizes?\s+(?P<comparison>(?:under|below|less\s+than|up\s+to|"
    rf"at\s+most|over|more\s+than|at\s+least)\s*(?:us\s*)?"
    rf"{MONEY_NUMBER_PATTERN})\b",
    re.IGNORECASE,
)

# Product features frequently contain phrases such as "up to 8-inch wrist"
# or "up to 100-hour chronograph".  Those are measurements, not prices, but
# the maximum-budget patterns intentionally accept amounts without a currency
# symbol (for example, "under 50").  Remove explicit measurements before
# looking for a budget so those feature descriptions cannot become accidental
# hard price limits.  Currency-prefixed quantities remain untouched.
NON_BUDGET_QUANTITY_RE = re.compile(
    rf"(?<![$£€\w]){MONEY_NUMBER_PATTERN}\s*(?:-\s*)?"
    r"(?:inch(?:es)?|in|feet|foot|ft|millimeters?|mm|centimeters?|cm|meters?|"
    r"hours?|hrs?|minutes?|mins?|seconds?|days?|weeks?|months?|years?|"
    r"ounces?|oz|pounds?|lbs?|grams?|kilograms?|kg|percent|%)"
    r"(?![a-z0-9])",
    re.IGNORECASE,
)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {_text(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_text(item) for item in value)
    return str(value)


def _size_marker(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.lower()).strip()
    normalized = SIZE_ALIASES.get(normalized, normalized)
    return "size_" + normalized.replace(".", "_").replace(" ", "_")


def _without_dimensions(text: str) -> str:
    """Mask complete dimensions while preserving offsets for size contexts."""

    return SIZE_DIMENSION_RE.sub(
        lambda match: " " * len(match.group(0)),
        text,
    )


def _scalar_texts(value: object) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _scalar_texts(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _scalar_texts(item)
        return
    yield str(value)


def _contextual_size_markers(value: object) -> list[str]:
    markers: list[str] = []
    for text in _scalar_texts(value):
        size_text = _without_dimensions(text)
        for context in SIZE_CONTEXT_RE.finditer(size_text):
            for match in SIZE_VALUE_RE.finditer(context.group("values")):
                value_start = context.start("values") + match.start()
                if SIZE_MEASUREMENT_RE.match(size_text[value_start:]):
                    continue
                markers.append(_size_marker(match.group(0)))
    return list(dict.fromkeys(markers))


def _structured_size_markers(details: object) -> list[str]:
    if not isinstance(details, dict):
        return []
    markers: list[str] = []
    for key, value in details.items():
        if str(key).strip().lower() != "size":
            continue
        for text in _scalar_texts(value):
            size_text = _without_dimensions(text)
            for match in SIZE_VALUE_RE.finditer(size_text):
                if SIZE_MEASUREMENT_RE.match(size_text[match.start():]):
                    continue
                markers.append(_size_marker(match.group(0)))
    return list(dict.fromkeys(markers))


def _attribute_tokens(attribute: object, value: object) -> list[str]:
    tokens = _tokens(value)
    if str(attribute).strip().lower() == "size":
        tokens.extend(_size_marker(match.group(0)) for match in SIZE_VALUE_RE.finditer(_text(value)))
    return list(dict.fromkeys(tokens))


def _tokens(value: object, *, contextual_sizes: bool = True) -> list[str]:
    ordinary = [
        token.lower()
        for token in TOKEN_RE.findall(_text(value))
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]
    size_markers = _contextual_size_markers(value) if contextual_sizes else []
    return list(dict.fromkeys([*ordinary, *size_markers]))


def _normalized_phrase(value: object) -> str:
    text = re.sub(r"n['’]t\b", " not", _text(value), flags=re.IGNORECASE)
    return " ".join(token.lower() for token in TOKEN_RE.findall(text))


def _corpus_segments(value: object) -> Iterable[str]:
    """Yield catalog scalars without letting negation leak across values."""

    if value is None:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            # Repeat the detail label for each nested leaf. This preserves its
            # meaning without flattening adjacent list values into one negation
            # scope (for example, ["No bleach", "Cotton"]).
            nested = list(_corpus_segments(item))
            if nested:
                for segment in nested:
                    yield f"{key} {segment}"
            else:
                yield str(key)
        return
    if isinstance(value, (list, tuple, set)):
        items = sorted(value, key=str) if isinstance(value, set) else value
        for item in items:
            yield from _corpus_segments(item)
        return
    yield str(value)


def _normalized_corpus(value: object) -> str:
    return SCALAR_BOUNDARY.join(
        segment
        for scalar in _corpus_segments(value)
        for clause in CLAUSE_BOUNDARY_RE.split(scalar)
        if (segment := _normalized_phrase(clause))
    )


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
        tokens = _tokens(value, contextual_sizes=field != "details")
        if field == "details":
            tokens.extend(_structured_size_markers(value))
        result[field] = (set(tokens), _normalized_corpus(value))
    return result


def _has_positive_occurrence(corpus: str, value: str) -> bool:
    """Return whether ``value`` occurs outside a local negative context."""

    pattern = re.compile(r"\b" + re.escape(value) + r"\b")
    for match in pattern.finditer(corpus):
        boundary_before = corpus.rfind(SCALAR_BOUNDARY, 0, match.start()) + 1
        boundary_after = corpus.find(SCALAR_BOUNDARY, match.end())
        if boundary_after < 0:
            boundary_after = len(corpus)
        prefix = TOKEN_RE.findall(
            corpus[max(boundary_before, match.start() - 60):match.start()]
        )[-3:]
        suffix = TOKEN_RE.findall(
            corpus[match.end():min(boundary_after, match.end() + 20)]
        )[:1]
        positive_not = len(prefix) >= 2 and prefix[-2:] in (
            ["not", "only"],
            ["not", "just"],
        )
        prefix_free = len(prefix) >= 2 and prefix[-2:] in (
            ["free", "of"],
            ["free", "from"],
        )
        if (
            (NEGATION_CONTEXT_TOKENS.intersection(prefix) or prefix_free)
            and not positive_not
        ) or suffix == ["free"]:
            continue
        return True
    return False


def _positive_field_weight(
    value: str,
    fields: dict[str, tuple[set[str], str]],
) -> float:
    """Return the strongest field weight for a non-negated token or phrase."""

    if value.startswith("size_"):
        return max(
            (
                weight
                for field, weight in FIELD_WEIGHTS.items()
                if value in fields[field][0]
            ),
            default=0.0,
        )
    return max(
        (
            weight
            for field, weight in FIELD_WEIGHTS.items()
            if _has_positive_occurrence(fields[field][1], value)
        ),
        default=0.0,
    )


def _weighted_match(
    query: object,
    fields: dict[str, tuple[set[str], str]],
    attribute: object = None,
) -> float:
    query_tokens = (
        _attribute_tokens(attribute, query)
        if attribute is not None
        else list(dict.fromkeys(_tokens(query)))
    )
    if not query_tokens:
        return 0.0

    token_total = sum(_positive_field_weight(token, fields) for token in query_tokens)
    token_score = token_total / len(query_tokens)

    phrases = {
        " ".join(query_tokens[index:index + size])
        for size in (2, 3)
        for index in range(len(query_tokens) - size + 1)
        if not any(
            token.startswith("size_")
            for token in query_tokens[index:index + size]
        )
    }
    if not phrases:
        return min(1.0, token_score)

    phrase_total = sum(_positive_field_weight(phrase, fields) for phrase in phrases)
    phrase_score = phrase_total / len(phrases)
    return min(1.0, 0.85 * token_score + 0.15 * phrase_score)


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        normalized = value.replace(",", "") if isinstance(value, str) else value
        number = float(normalized)
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


def _constraint_entries(active_constraints: object) -> list[tuple[str, str, int]]:
    if not isinstance(active_constraints, dict):
        return []
    values: list[tuple[str, str, int]] = []
    for attribute, items in active_constraints.items():
        normalized_attribute = str(attribute).strip().lower()
        if normalized_attribute == "budget":
            continue
        if isinstance(items, (str, int, float)):
            items = [items]
        if not isinstance(items, Iterable) or isinstance(items, dict):
            continue
        for index, item in enumerate(items):
            rendered = _text(item).strip()
            if rendered:
                values.append((normalized_attribute, rendered, index))
    return values


def _constraint_payload(value: object) -> str:
    return CONSTRAINT_LABEL_RE.sub("", _text(value).strip()).strip()


def _mapping_value(mapping: object, key: object) -> object:
    if not isinstance(mapping, dict):
        return None
    normalized_key = _normalized_phrase(key)
    for candidate_key, candidate_value in mapping.items():
        if _normalized_phrase(candidate_key) == normalized_key:
            return candidate_value
    return None


def _priority_spec(
    priorities: object,
    attribute: str,
    value: str,
    value_index: int,
) -> object:
    """Resolve flexible attribute-, value-, or record-level priority metadata."""

    if isinstance(priorities, dict):
        attribute_spec = _mapping_value(priorities, attribute)
        if attribute_spec is None:
            return _mapping_value(priorities, value)
        if isinstance(attribute_spec, dict):
            value_spec = _mapping_value(attribute_spec, value)
            if value_spec is not None:
                return value_spec
            return (
                _mapping_value(attribute_spec, "priority")
                or _mapping_value(attribute_spec, "provenance")
                or _mapping_value(attribute_spec, "source")
            )
        if isinstance(attribute_spec, (list, tuple)):
            return attribute_spec[value_index] if value_index < len(attribute_spec) else None
        return attribute_spec

    if isinstance(priorities, (list, tuple)):
        for record in priorities:
            if not isinstance(record, dict):
                continue
            if (
                _normalized_phrase(_mapping_value(record, "attribute"))
                == _normalized_phrase(attribute)
                and _normalized_phrase(_mapping_value(record, "value"))
                == _normalized_phrase(value)
            ):
                return (
                    _mapping_value(record, "priority")
                    or _mapping_value(record, "provenance")
                    or _mapping_value(record, "source")
                )
    return None


def _priority_weight(spec: object) -> float:
    if isinstance(spec, bool):
        return CONSTRAINT_PRIORITY_WEIGHTS["normal"]
    if isinstance(spec, (int, float)):
        return max(0.1, min(10.0, _number(spec, 1.0)))
    normalized = _normalized_phrase(spec).replace(" ", "_")
    if normalized in HARD_PRIORITY_NAMES:
        return CONSTRAINT_PRIORITY_WEIGHTS["hard"]
    if normalized in SOFT_PRIORITY_NAMES:
        return CONSTRAINT_PRIORITY_WEIGHTS["soft"]
    return CONSTRAINT_PRIORITY_WEIGHTS["normal"]


def _is_hard_priority(spec: object) -> bool:
    if isinstance(spec, bool):
        return False
    if isinstance(spec, (int, float)):
        return _number(spec, 1.0) > CONSTRAINT_PRIORITY_WEIGHTS["normal"]
    return _normalized_phrase(spec).replace(" ", "_") in HARD_PRIORITY_NAMES


def _known_match_score(
    attribute: object,
    value: object,
    fields: dict[str, tuple[set[str], str]],
) -> float:
    """Return a conservative, unnegated catalog match for an excluded value."""

    tokens = _attribute_tokens(attribute, _constraint_payload(value))
    if not tokens:
        return 0.0

    # Size markers are deliberately synthetic (for example ``size_s`` and
    # ``size_8_5``), so they are present in each field's token set but not in
    # its normalized prose corpus.  Check those exact markers first; otherwise
    # an explicit exclusion such as "not size S" would never be enforced.
    size_markers = {token for token in tokens if token.startswith("size_")}
    if size_markers:
        return max(
            (
                weight
                for field, weight in FIELD_WEIGHTS.items()
                if size_markers.issubset(fields[field][0])
            ),
            default=0.0,
        )

    phrase = " ".join(tokens)
    strongest = 0.0
    for field, weight in FIELD_WEIGHTS.items():
        corpus = fields[field][1]
        if _has_positive_occurrence(corpus, phrase):
            strongest = max(strongest, weight)
        if strongest >= weight:
            continue

        # Catalog wording often inserts useful qualifiers between the user's
        # words ("genuine cowhide leather" versus "genuine leather").  Treat
        # all independently positive token mentions in one field as known
        # evidence, while retaining the same local negation checks.
        all_positive = True
        for token in tokens:
            if not _has_positive_occurrence(corpus, token):
                all_positive = False
                break
        if all_positive:
            strongest = max(strongest, weight)
    return strongest


def _excluded_match_score(
    excluded_constraints: object,
    fields: dict[str, tuple[set[str], str]],
) -> float:
    return max(
        (
            _known_match_score(attribute, value, fields)
            for attribute, value, _ in _constraint_entries(excluded_constraints)
        ),
        default=0.0,
    )


def _budget_values(active_constraints: object) -> list[str]:
    if not isinstance(active_constraints, dict):
        return []
    budget = active_constraints.get("budget", [])
    if isinstance(budget, (str, int, float)):
        budget = [budget]
    if not isinstance(budget, Iterable) or isinstance(budget, dict):
        return []
    return [_text(item).strip() for item in budget if _text(item).strip()]


def _first_budget_number(
    patterns: Iterable[re.Pattern],
    text: str,
    *,
    highest: bool = False,
) -> float | None:
    matches: list[float] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            value = _number(match.group(1), default=-1.0)
            if value >= 0.0:
                matches.append(value)
    if not matches:
        return None
    return max(matches) if highest else min(matches)


def _parsed_budget_preference(
    text: str,
    range_patterns: Iterable[re.Pattern],
) -> tuple[str, float, float] | None:
    for pattern in range_patterns:
        match = pattern.search(text)
        if match:
            first = _number(match.group(1), default=-1.0)
            second = _number(match.group(2), default=-1.0)
            if first >= 0.0 and second >= 0.0:
                return "range", min(first, second), max(first, second)

    target = _first_budget_number(BUDGET_TARGET_PATTERNS, text)
    if target is not None:
        return "target", target, target

    maximum = _first_budget_number(MAX_BUDGET_PATTERNS, text)
    if maximum is not None:
        return "maximum", 0.0, maximum

    # Multiple minimum requirements form a conjunction, so retain the
    # strongest (largest) lower bound rather than the smallest number.
    minimum = _first_budget_number(MIN_BUDGET_PATTERNS, text, highest=True)
    if minimum is not None:
        return "minimum", minimum, math.inf
    return None


def _budget_preference(
    active_constraints: object,
    user_message: str,
) -> tuple[str, float, float] | None:
    """Return ``(kind, lower, upper)`` for a contextual price preference.

    A bare number is interpreted as a maximum only when it came through the
    structured ``budget`` slot. This preserves answers such as ``$50`` to the
    agent's explicit maximum-budget question without mistaking a shoe size or
    product measurement in ordinary prose for a price.
    """

    size_comparisons = list(NON_BUDGET_SIZE_COMPARISON_RE.finditer(user_message))
    ignored_budget_values = {
        _normalized_phrase(match.group("comparison"))
        for match in size_comparisons
    }
    message_text = NON_BUDGET_SIZE_COMPARISON_RE.sub(" ", user_message)
    budget_values = [
        value
        for value in _budget_values(active_constraints)
        if _normalized_phrase(value) not in ignored_budget_values
    ]

    # Current-message evidence takes precedence over stored values. Besides
    # respecting overrides, this recovers the original ``$1,000`` when an
    # upstream conservative parser retained only ``under $1``.
    cleaned_message = NON_BUDGET_QUANTITY_RE.sub(" ", message_text)
    preference = _parsed_budget_preference(cleaned_message, BUDGET_RANGE_PATTERNS)
    if preference is not None:
        return preference

    # Values already stored in the budget slot have explicit context, so a
    # compact range such as ``50 to 100`` is safe to interpret as money. Keep
    # this separate from free-form prose so shoe sizes and dimensions do not
    # become accidental price filters.
    cleaned_values = [
        NON_BUDGET_QUANTITY_RE.sub(" ", value)
        for value in budget_values
    ]
    preference = _parsed_budget_preference(
        " ".join(cleaned_values),
        STRUCTURED_BUDGET_RANGE_PATTERNS,
    )
    if preference is not None:
        return preference

    # Pending budget answers arrive in active_constraints. A bare value in the
    # free-form message alone remains deliberately ambiguous.
    for cleaned in reversed(cleaned_values):
        numbers = CONTEXTUAL_BUDGET_NUMBER_RE.findall(cleaned)
        if len(numbers) == 1:
            maximum = _number(numbers[0], default=-1.0)
            if maximum >= 0.0:
                return "maximum", 0.0, maximum
    return None


def _budget_distance(price: float, preference: tuple[str, float, float]) -> float:
    kind, lower, upper = preference
    if kind == "target":
        return abs(price - lower) / max(lower, 1.0)
    if price < lower:
        return (lower - price) / max(lower, 1.0)
    if price > upper:
        reference = upper if math.isfinite(upper) else lower
        return (price - upper) / max(reference, 1.0)
    return 0.0


def _constraint_score(
    active_constraints: object,
    fields: dict[str, tuple[set[str], str]],
    constraint_priorities: object = None,
) -> float:
    entries = _constraint_entries(active_constraints)
    if not entries:
        return 0.0
    weighted_total = 0.0
    total_weight = 0.0
    strongest_hard_match = 0.0
    for attribute, value, index in entries:
        priority = _priority_spec(constraint_priorities, attribute, value, index)
        weight = _priority_weight(priority)
        match_score = _weighted_match(value, fields, attribute)
        weighted_total += weight * match_score
        total_weight += weight
        if _is_hard_priority(priority):
            strongest_hard_match = max(strongest_hard_match, match_score)
    if not total_weight:
        return 0.0
    # A known match for an explicit requirement deserves more than a modest
    # averaging weight: otherwise several incidental soft-preference matches
    # can collectively outrank the user's must-have.  Missing evidence still
    # receives no penalty because catalog metadata is incomplete.
    return min(
        1.0,
        weighted_total / total_weight
        + HARD_CONSTRAINT_MATCH_BOOST * strongest_hard_match,
    )


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
    *,
    excluded_constraints: dict[str, list[str]] | None = None,
    constraint_priorities: object = None,
) -> list[dict]:
    """Return deterministic, ranked copies of retrieval candidates.

    Each candidate should contain ``parent_asin``, a nested ``product`` catalog
    record, a normalized ``retrieval_score`` (larger is better), and optional
    ``route_hits``. Incomplete candidates are tolerated and inputs are not mutated.
    Optional exclusions use conservative all-token matching: known violations are
    placed behind non-violating alternatives but remain as a last-resort fallback.
    Priority metadata may label constraints ``hard`` or ``soft`` by attribute,
    by value, with parallel lists, or with provenance records.
    """

    try:
        limit = max(0, int(top_k))
    except (TypeError, ValueError):
        return []
    if limit == 0:
        return []

    # 1. A product can arrive through several routes. Merge route evidence and
    # keep one immutable candidate per parent_asin.
    unique = _deduplicate(candidates)
    if not unique:
        return []

    max_log_ratings = max(
        (math.log1p(max(0.0, _number(_product(item).get("rating_number")))) for item in unique),
        default=0.0,
    )

    # 2. Compute the shared relevance formula for every candidate.
    scored: list[tuple[float, str, dict, float | None, float]] = []
    for candidate in unique:
        parent_asin = str(candidate["parent_asin"])
        product = _product(candidate)
        fields = _field_corpora(product)
        score = (
            SCORE_WEIGHTS["retrieval"] * _unit_score(candidate.get("retrieval_score"))
            + SCORE_WEIGHTS["message"] * _weighted_match(user_message, fields)
            + SCORE_WEIGHTS["constraints"] * _constraint_score(
                active_constraints,
                fields,
                constraint_priorities,
            )
            + SCORE_WEIGHTS["routes"] * _route_score(candidate)
            + SCORE_WEIGHTS["profile"] * _profile_score(user_profile, fields)
            + SCORE_WEIGHTS["quality"] * _quality_score(product, max_log_ratings)
        )
        scored.append((
            score,
            parent_asin,
            candidate,
            _price(product),
            _excluded_match_score(excluded_constraints, fields),
        ))

    def relevance_key(
        item: tuple[float, str, dict, float | None, float],
    ) -> tuple[float, str]:
        return (-item[0], item[1])

    # 3. Apply budget policy after relevance: verified in-budget products lead,
    # unknown prices remain eligible, and over-budget items are fallback only.
    def order_group(
        items: list[tuple[float, str, dict, float | None, float]],
    ) -> list[tuple[float, str, dict, float | None, float]]:
        budget = _budget_preference(active_constraints, user_message)
        if budget is None:
            return sorted(items, key=relevance_key)

        if budget[0] == "target":
            # "Around $50" is a soft proximity preference, not a hard filter.
            # Cap its influence so a close but irrelevant item cannot overtake
            # a substantially stronger text match. Missing prices remain fully
            # eligible with a modest uncertainty penalty.
            return sorted(
                items,
                key=lambda item: (
                    -(
                        item[0]
                        - (0.05 if item[3] is None else min(
                            0.15, 0.15 * _budget_distance(item[3], budget)
                        ))
                    ),
                    item[1],
                ),
            )

        within_budget = [
            item
            for item in items
            if item[3] is not None and _budget_distance(item[3], budget) == 0.0
        ]
        unknown_price = [item for item in items if item[3] is None]
        over_budget = [
            item
            for item in items
            if item[3] is not None and _budget_distance(item[3], budget) > 0.0
        ]

        within_budget.sort(key=relevance_key)
        # Missing-price products remain eligible but follow verified in-budget products.
        unknown_price.sort(key=relevance_key)
        # When fallback is necessary, prefer the smallest budget violation first.
        over_budget.sort(
            key=lambda item: (
                _budget_distance(item[3], budget),
                *relevance_key(item),
            )
        )

        viable = [*within_budget, *unknown_price]
        return viable if len(viable) >= 10 else [*viable, *over_budget]

    # 4. Known explicit-negative violations trail compliant/unknown products;
    # missing catalog evidence is never treated as proof of a mismatch.
    non_violating = order_group([item for item in scored if item[4] == 0.0])
    known_violations = order_group([item for item in scored if item[4] > 0.0])
    ordered = (
        non_violating
        if len(non_violating) >= limit
        else [*non_violating, *known_violations]
    )
    return [item[2] for item in ordered[:limit]]
