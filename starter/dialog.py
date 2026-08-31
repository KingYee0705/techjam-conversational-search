"""Deterministic conversation memory and clarification strategy.

This module turns each customer message into the structured ``decision`` used
by both retrieval routes and the ranker. It keeps active preferences separate
from superseded override values, explicit negatives, and declined attributes.
"""

from __future__ import annotations

import copy
import re


ATTRIBUTE_ORDER = (
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
)
ALLOWED_ATTRIBUTES = set(ATTRIBUTE_ORDER)

MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex",
    "silk", "rayon", "fabric", "linen", "cashmere", "suede", "velvet",
    "rubber", "acrylic", "denim", "fleece", "canvas", "satin", "mesh",
)
COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown",
    "gray", "grey", "purple", "yellow", "orange", "navy", "beige",
    "gold", "silver", "teal", "maroon", "khaki", "tan", "cream",
    "burgundy",
)
CATEGORY_TERMS = {
    "accessory", "accessories", "bag", "bags", "belt", "belts", "boot",
    "boots", "bracelet", "bracelets", "clothing", "coat", "coats", "dress",
    "dresses", "earring", "earrings", "footwear", "handbag", "handbags",
    "hat", "hats", "jacket", "jackets", "jeans", "jewelry", "necklace",
    "necklaces", "pants", "purse", "purses", "ring", "rings", "sandal",
    "sandals", "shirt", "shirts", "shoe", "shoes", "shorts", "skirt",
    "skirts", "sneaker", "sneakers", "sock", "socks", "sweater", "sweaters",
    "swimwear", "t-shirt", "tee", "tees", "top", "tops", "trousers", "watch",
    "watches",
}

CATEGORY_RE = re.compile(
    r"\b(?:i(?:'m| am)\s+)?looking\s+for\s+(.+?)"
    r"(?=\s+(?:but|except|without)\b|[.,;]|$)",
    re.IGNORECASE,
)
REQUIREMENT_RE = re.compile(
    r"\ba\s+key\s+requirement\s+is\s*:\s*(.+?)\s*\.?\s*$",
    re.IGNORECASE,
)
MATTERS_RE = re.compile(
    r"\bwhat\s+matters\s+is\s*:\s*(.+?)\s*\.?\s*$",
    re.IGNORECASE,
)
OVERRIDE_RE = re.compile(
    r"\bwhat\s+i\s+need\s+is\s*:\s*(.+?)\s*\.?\s*$",
    re.IGNORECASE,
)
NO_ADDITIONAL_PREFERENCE_RE = re.compile(
    r"\b(?:(?:i\s+)?(?:(?:don't|do\s+not)\s+have|have\s+no)|no)\s+"
    r"(?:an?\s+)?additional\s+preference"
    r"(?:\s+(?:for|on|about)\s+([a-z_ -]+?))?(?=[;,.!?]|$)",
    re.IGNORECASE,
)
NO_PREFERENCE_RE = re.compile(
    r"\b(?:i\s+)?(?:(?:don't|do\s+not)\s+have\s+(?:an?\s+)?"
    r"(?:(?:particular|strong|specific)\s+)?preference|have\s+no\s+"
    r"(?:(?:particular|strong|specific)\s+)?preference|no\s+"
    r"(?:(?:particular|strong|specific)\s+)?preference)"
    r"(?:\s+(?:for|on|about|regarding|with\s+respect\s+to|"
    r"when\s+it\s+comes\s+to)\s+([a-z_ -]+?))?(?=[;,.!?]|$)",
    re.IGNORECASE,
)
DONT_MIND_RE = re.compile(
    r"\b(?:i\s+)?(?:don't|do\s+not)\s+(?:mind|care(?:\s+about)?)"
    r"(?:\s+(?:the\s+)?(.+?))?\s*[.!?]?$",
    re.IGNORECASE,
)
ANY_ATTRIBUTE_FINE_RE = re.compile(
    r"\b(?:any|either)\s+(.+?)\s+(?:is|would\s+be|will\s+be|works?)\s+"
    r"(?:fine|okay|ok|acceptable)\s*[.!?]?$",
    re.IGNORECASE,
)
ANY_ATTRIBUTE_WORKS_RE = re.compile(
    r"\b(?:any|either)\s+(.+?)\s+(?:works?(?:\s+for\s+me)?|will\s+do|"
    r"is\s+(?:fine|okay|ok|acceptable))\s*[.!?]?$",
    re.IGNORECASE,
)
OPEN_TO_ANY_ATTRIBUTE_RE = re.compile(
    r"\b(?:i(?:'m|\s+am)\s+)?(?:open\s+to|okay\s+with|ok\s+with|"
    r"fine\s+with|happy\s+with)\s+(?:any|either)\s+(.+?)\s*[.!?]?$",
    re.IGNORECASE,
)
NOT_PICKY_RE = re.compile(
    r"\b(?:i(?:'m|\s+am)\s+)?not\s+(?:picky|fussy)\s+"
    r"(?:about|on|with)\s+(?:the\s+)?(.+?)\s*[.!?]?$",
    re.IGNORECASE,
)
ATTRIBUTE_UP_TO_YOU_RE = re.compile(
    r"\b(?:the\s+)?(.+?)\s+(?:is\s+)?up\s+to\s+you\s*[.!?]?$",
    re.IGNORECASE,
)
ATTRIBUTE_DOES_NOT_MATTER_RE = re.compile(
    r"\b(?:the\s+)?(.+?)\s+(?:doesn't|does\s+not)\s+matter"
    r"(?:\s+to\s+me)?\s*[.!?]?$",
    re.IGNORECASE,
)
FLEXIBLE_ATTRIBUTE_RE = re.compile(
    r"\b(?:i(?:'m|\s+am)\s+)?flexible\s+(?:about|on|with)\s+"
    r"(?:the\s+)?(.+?)\s*[.!?]?$",
    re.IGNORECASE,
)
WHATEVER_ATTRIBUTE_RE = re.compile(
    r"\bwhatever\s+(.+?)\s*[.!?]?$",
    re.IGNORECASE,
)
GENERIC_DECLINE_RE = re.compile(
    r"^\s*(?:no\s+(?:(?:particular|additional)\s+)?preference|either\s+is\s+fine|"
    r"anything\s+(?:is\s+fine|works)|whatever\s+works|"
    r"i(?:'m|\s+am)\s+flexible)\s*[.!?]?\s*$",
    re.IGNORECASE,
)
DECLINE_EXCEPTION_RE = re.compile(
    r"\b(?:except|unless|as\s+long\s+as|provided(?:\s+that)?|only\s+if)\b",
    re.IGNORECASE,
)
OVERRIDE_SIGNAL_RE = re.compile(
    r"\b(?:actually|instead|ignore\s+(?:my\s+)?earlier|changed?\s+my\s+mind|"
    r"rather\s+than|(?:^|(?:actually|please)\s*,?\s+)make\s+it|"
    r"what\s+i\s+need\s+is|switch(?:ing)?\s+"
    r"(?:from\s+.+?\s+)?to|replace\s+.+?\s+with|no\s+longer|"
    r"now\s+i\s+(?:want|prefer|need))\b",
    re.IGNORECASE,
)
GENERIC_REJECTION_RE = re.compile(
    r"\b(?:options\s+are\s+not\s+quite\s+right|ask\s+me\s+about\s+one\s+specific)\b",
    re.IGNORECASE,
)
NEGATIVE_VALUE_RE = re.compile(
    r"\b(?:(?:not|without|avoid(?:ing)?|except|dislike)\s+|"
    r"(?:don't|do\s+not)\s+(?:want|like)\s+|anything\s+but\s+|"
    r"(?:any|either)\s+(?:color|colour|material|fabric|style|brand|size)\s+"
    r"(?:is\s+fine\s*,?\s+)?but\s+"
    r"(?!(?:(?:please\s+)?(?:show|give|list|recommend)\b|"
    r"i\s+(?:want|need|prefer)\b|make\s+it\b))|"
    r"steer\s+clear\s+of\s+|no\s+(?!more\b|additional\b|particular\b|"
    r"strong\b|specific\b|preference\b))"
    r"([a-z][a-z0-9 -]{0,60}?)(?=\s+please\b|[,.;]|\s+but\b|$)",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

QUESTION_ORDER = (
    "feature", "material", "color", "style", "size", "use_case", "budget", "brand",
)
QUESTION_TEXT = {
    "category": "What type of clothing, shoes, or accessory are you looking for?",
    "feature": "Is there a particular feature you want me to prioritize?",
    "material": "Do you have a material preference?",
    "color": "Do you have a color preference?",
    "style": "What style or fit would you prefer?",
    "size": "Do you have a size or width requirement?",
    "use_case": "What occasion or activity will you use it for?",
    "budget": "What is the maximum budget you would like me to use?",
    "brand": "Do you have a preferred brand?",
    "other": "Is there another requirement that would help narrow these down?",
}


def _clean(value: object, limit: int = 240) -> str:
    if value is None:
        return ""
    rendered = str(value).translate(str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'"}))
    return re.sub(r"\s+", " ", rendered).strip(" \t\n.,;")[:limit].rstrip()


def _key(value: object) -> str:
    return " ".join(token.lower() for token in TOKEN_RE.findall(_clean(value)))


def classify_attribute(value: str) -> str:
    """Classify a preference using the evaluator's public attribute rules."""

    lowered = value.lower()
    size_context = re.search(
        r"\b(?:size|sizing|width|wide|narrow)\b", lowered
    )
    monetary_context = re.search(
        r"(?:[$£€]|\busd\b|\b(?:budget|price|cost|spend(?:ing)?)\b)",
        lowered,
    )
    if size_context and not monetary_context:
        return "size"
    if (
        any(word in lowered for word in ("budget", "price", "cost", "spend"))
        and re.search(r"\d", lowered)
    ) or re.search(
        r"(?:[$£€]|\busd\b|<=|\b(?:under|below|at most|no more than)\s+)\s*\d",
        lowered,
    ):
        return "budget"
    if any(
        re.search(rf"\b{re.escape(material)}\b", lowered)
        for material in MATERIALS
    ):
        return "material"
    if "color" in lowered or any(
        re.search(rf"\b{re.escape(color)}\b", lowered) for color in COLORS
    ):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


def _attribute_from_text(value: str) -> str | None:
    normalized = value.lower().replace("-", "_").replace(" ", "_").strip(" _")
    if normalized in ALLOWED_ATTRIBUTES:
        return normalized
    words = set(TOKEN_RE.findall(value.lower()))
    aliases = {
        "budget": {"price", "cost", "spend", "spending"},
        "material": {"fabric", "textile"},
        "color": {"colour", "hue", "shade"},
        "style": {"fit", "look"},
        "use_case": {"activity", "occasion", "purpose", "use"},
    }
    for attribute, terms in aliases.items():
        if words.intersection(terms):
            return attribute
    for attribute in ATTRIBUTE_ORDER:
        if attribute == "use_case":
            if {"use", "case"}.issubset(words):
                return attribute
        elif attribute in words:
            return attribute
    return None


def _decline_target(value: object, pending: str | None) -> str | None:
    """Resolve an explicit attribute, using pending context only for pronouns."""

    cleaned = _clean(value)
    generic = _key(cleaned)
    if not generic or generic in {
        "it", "that", "either", "anything", "whatever", "which", "which one",
        "works",
    }:
        return pending
    words = set(TOKEN_RE.findall(cleaned.lower()))
    fillers = {
        "a", "an", "any", "either", "item", "my", "of", "preference",
        "product", "the",
    }
    substantive = words - fillers
    decline_words = {
        "category": {"category", "type", "item", "product"},
        "material": {"material", "fabric", "textile"},
        "color": {"color", "colour", "hue", "shade"},
        "size": {"shoe", "shoes", "size", "sizing", "width"},
        "style": {"style", "fit", "look"},
        "brand": {"brand", "maker", "name"},
        "budget": {"budget", "price", "cost", "spend", "spending"},
        "feature": {"feature", "features"},
        "use_case": {"use", "case", "activity", "occasion", "purpose"},
        "other": {"other", "detail", "requirement"},
    }
    for attribute, allowed in decline_words.items():
        if substantive and substantive.issubset(allowed):
            return attribute
    return None


def _split_values(value: str) -> list[str]:
    return [cleaned for item in value.split(";") if (cleaned := _clean(item))]


def _negative_payload(value: str) -> str | None:
    match = re.match(
        r"^(?:i\s+)?(?:(?:not|without|avoid(?:ing)?|except|dislike)\s+|"
        r"(?:don't|do\s+not)\s+(?:want|like)\s+|anything\s+but\s+|"
        r"steer\s+clear\s+of\s+|no\s+(?!more\b|additional\b|particular\b|"
        r"strong\b|specific\b|preference\b))"
        r"(.+?)(?:\s+please)?$",
        _clean(value),
        re.IGNORECASE,
    )
    return _clean(match.group(1)) if match else None


def _split_negative_values(value: object) -> list[str]:
    """Split explicit alternatives while preserving multi-word phrases."""

    return [
        cleaned
        for item in re.split(
            r"\s+(?:and|or)\s+|[,;]", _clean(value), flags=re.IGNORECASE
        )
        if (
            cleaned := _clean(
                re.sub(
                    r"^(?:(?:any|the)\s+|made\s+(?:of|from|with)\s+)+",
                    "",
                    item,
                    flags=re.IGNORECASE,
                )
            )
        )
    ]


def _negative_payloads(value: str) -> list[str]:
    payload = _negative_payload(value)
    return _split_negative_values(payload) if payload else []


def _positive_residuals(
    message: str,
    negative_matches: list[re.Match],
) -> list[str]:
    """Return positive clauses outside recognized negative spans."""

    segments: list[str] = []
    cursor = 0
    for match in negative_matches:
        segments.append(message[cursor:match.start()])
        cursor = match.end()
    segments.append(message[cursor:])

    values: list[str] = []
    generic = {"and", "anything", "but", "either", "i", "please"}
    for segment in segments:
        for clause in re.split(r"[.,;!?]|\bbut\b", segment, flags=re.IGNORECASE):
            cleaned = _clean(re.sub(r"^(?:and|also|but)\s+", "", clause, flags=re.IGNORECASE))
            if (
                not cleaned
                or _key(cleaned) in generic
                or CATEGORY_RE.search(cleaned)
                or _negative_payload(cleaned)
                or NEGATIVE_VALUE_RE.search(cleaned)
            ):
                continue
            values.append(cleaned)
    return list(dict.fromkeys(values))


def _positive_answer_values(message: str) -> list[str]:
    """Preserve qualifiers while separating clearly coordinated answers."""

    requested_options = re.search(
        r"\bbut\s+(?:please\s+)?(?:show|give|list|recommend)\s+(?:me\s+)?"
        r"(.+?)\s+(?:options|choices|recommendations)\b",
        message,
        re.IGNORECASE,
    )
    if requested_options:
        return [_clean(requested_options.group(1))]
    concrete_after_indifference = re.search(
        r"\b(?:don't|do\s+not)\s+mind\s+(?:the\s+)?(?:color|colour|material|"
        r"fabric|style|brand|size)\s+(?:being|as)\s+(.+?)\s*[.!?]?\s*$",
        message,
        re.IGNORECASE,
    )
    if concrete_after_indifference:
        return [_clean(concrete_after_indifference.group(1))]
    values = [
        cleaned
        for item in re.split(r"\s+(?:and|also)\s+|;", message, flags=re.IGNORECASE)
        if (cleaned := _clean(item))
    ]
    return values or [_clean(message)]


def _looks_like_category(value: str) -> bool:
    tokens = TOKEN_RE.findall(value.lower())
    return len(tokens) <= 7 and any(token in CATEGORY_TERMS for token in tokens)


def _direct_override(message: str) -> tuple[list[str], list[str]]:
    """Return (replacement values, explicitly superseded values)."""

    switch = re.search(
        r"\bswitch(?:ing)?(?:\s+from\s+(.+?))?\s+to\s+(.+?)\s*[.!?]?\s*$",
        message,
        re.IGNORECASE,
    )
    if switch:
        old = [_clean(switch.group(1))] if switch.group(1) else []
        return [_clean(switch.group(2))], old

    replace = re.search(
        r"\breplace\s+(.+?)\s+with\s+(.+?)\s*[.!?]?\s*$",
        message,
        re.IGNORECASE,
    )
    if replace:
        return [_clean(replace.group(2))], [_clean(replace.group(1))]

    no_longer_with_replacement = re.search(
        r"\bno\s+longer\s+(?:want|need|prefer|like)?\s*(.+?)\s*"
        r"(?:[;,]|\s+but\s+|\s+instead\s+)\s*"
        r"(?:now\s+)?(?:i\s+(?:want|need|prefer)\s+)?(.+?)\s*[.!?]?\s*$",
        message,
        re.IGNORECASE,
    )
    if no_longer_with_replacement:
        replacement = re.sub(
            r"\s+please$", "", _clean(no_longer_with_replacement.group(2)),
            flags=re.IGNORECASE,
        )
        return [replacement], [_clean(no_longer_with_replacement.group(1))]

    no_longer = re.search(
        r"\bno\s+longer\s+(?:want|need|prefer|like)?\s*(.+?)\s*[.!?]?\s*$",
        message,
        re.IGNORECASE,
    )
    if no_longer:
        return [], [_clean(no_longer.group(1))]

    now = re.search(
        r"\bnow\s+i\s+(?:want|prefer|need)\s+(.+?)\s*[.!?]?\s*$",
        message,
        re.IGNORECASE,
    )
    if now:
        return [_clean(now.group(1))], []

    instead_of = re.search(
        r"(?:actually\s*,?\s*)?(?:i\s+(?:need|want)\s+|make\s+it\s+)?"
        r"(.+?)\s+instead\s+of\s+(.+?)\s*\.?\s*$",
        message,
        re.IGNORECASE,
    )
    if instead_of:
        return [_clean(instead_of.group(1))], [_clean(instead_of.group(2))]

    # Parse the value after "make it" before the generic trailing "instead"
    # form, which would otherwise retain the instruction words themselves.
    make_it = re.search(
        r"\bmake\s+it\s+(.+?)(?:\s+instead)?\s*\.?\s*$",
        message,
        re.IGNORECASE,
    )
    if make_it:
        return [_clean(make_it.group(1))], []

    leading_instead = re.search(
        r"^\s*instead\s*[,;:]?\s*(?:i\s+(?:want|need|prefer)\s+)?"
        r"(.+?)\s*[.!?]?\s*$",
        message,
        re.IGNORECASE,
    )
    if leading_instead:
        return [_clean(leading_instead.group(1))], []

    trailing_instead = re.search(
        r"^(?:i\s+(?:want|need|prefer)\s+)?(.+?)\s+instead\s*[.!?]?\s*$",
        message,
        re.IGNORECASE,
    )
    if trailing_instead:
        return [_clean(trailing_instead.group(1))], []

    rather_than = re.search(
        r"(?:actually\s*,?\s*)?(?:i(?:'m| am)\s+looking\s+for\s+|"
        r"i\s+(?:need|want)\s+|make\s+it\s+)?"
        r"(.+?)\s+rather\s+than\s+(.+?)\s*\.?\s*$",
        message,
        re.IGNORECASE,
    )
    if rather_than:
        return [_clean(rather_than.group(1))], [_clean(rather_than.group(2))]

    not_but = re.search(
        r"\bnot\s+(.+?)(?:,|\s+but\s+)(.+?)(?:\s+instead)?\s*\.?\s*$",
        message,
        re.IGNORECASE,
    )
    if not_but:
        return [_clean(not_but.group(2))], [_clean(not_but.group(1))]

    changed_mind = re.search(
        r"\bchanged?\s+my\s+mind\s*[,;:]?\s*"
        r"(?:i\s+(?:need|want|prefer)\s+|make\s+it\s+)?(.+?)\s*\.?\s*$",
        message,
        re.IGNORECASE,
    )
    if changed_mind:
        return [_clean(changed_mind.group(1))], []

    actually = re.search(
        r"\bactually\s*,?\s*(?:i\s+(?:need|want|prefer)\s+)?(.+?)\s*\.?\s*$",
        message,
        re.IGNORECASE,
    )
    if actually and "ignore" not in actually.group(1).lower():
        return [_clean(actually.group(1))], []
    return [], []


def _obvious_constraints(message: str) -> list[str]:
    """Extract conservative structured values from free-form messages."""

    values: list[str] = []
    nonmonetary_size = bool(
        re.search(r"\b(?:size|sizing|width|wide|narrow)\b", message, re.IGNORECASE)
        and not re.search(
            r"(?:[$£€]|\busd\b|\b(?:budget|price|cost|spend(?:ing)?)\b)",
            message,
            re.IGNORECASE,
        )
    )
    amount = r"(?:usd\s*)?[$£€]?\s*\d[\d,]*(?:\.\d+)?"
    currency_amount = r"(?:(?:usd)\s*|[$£€]\s*)\d[\d,]*(?:\.\d+)?"
    budget_patterns = (
        # Maximums may omit a currency marker because the limiting language is
        # already unambiguously price-like in ordinary shopping requests.
        rf"\b(?:under|below|less\s+than|up\s+to|at\s+most|no\s+more\s+than|"
        rf"maximum(?:\s+(?:budget\s+)?of)?)\s*{amount}\b",
        # Explicitly labelled ranges may use bare numbers.
        rf"\b(?:budget|price|cost|spend(?:ing)?)\b.{{0,20}}?"
        rf"\b(?:between|from)\s+{amount}\s*(?:and|to|[-–—])\s*{amount}\b",
        rf"\b(?:budget|price|cost|spend(?:ing)?)\b\s*(?:is|of|:)?\s*"
        rf"{amount}\s*(?:to|[-–—])\s*{amount}\b",
        # Unlabelled ranges are accepted only when a currency marker makes the
        # monetary meaning explicit (and avoids treating shoe-size ranges as budgets).
        rf"\b(?:between|from)\s+{currency_amount}\s*"
        rf"(?:and|to|[-–—])\s*{amount}\b",
        rf"{currency_amount}\s*(?:to|[-–—])\s*{amount}\b",
        # Likewise, a minimum needs either a currency marker or a budget label.
        rf"\b(?:at\s+least|over|above|more\s+than|minimum(?:\s+(?:budget\s+)?of)?)"
        rf"\s*{currency_amount}\b",
        rf"\b(?:budget|price|cost|spend(?:ing)?)\b.{{0,20}}?"
        rf"\b(?:at\s+least|over|above|more\s+than|minimum(?:\s+of)?)\s*{amount}\b",
        rf"\bminimum\s+budget(?:\s+of)?\s*{amount}\b",
        # Finally retain an explicitly labelled single amount, including commas.
        rf"\b(?:budget|price|cost)\b\s*(?:is|of|:)?\s*{amount}\b",
    )
    if not nonmonetary_size:
        for pattern in budget_patterns:
            budget = re.search(pattern, message, re.IGNORECASE)
            if budget:
                values.append(_clean(budget.group(0)))
                break
    lowered = message.lower()
    for term in (*MATERIALS, *COLORS):
        is_negative = re.search(
            rf"\b(?:(?:not|without|avoid(?:ing)?|except|no)\s+|"
            rf"(?:don't|do\s+not)\s+(?:want|like)\s+|anything\s+but\s+|"
            rf"(?:any|either)\s+(?:color|colour|material|fabric|style|brand|size)\s+"
            rf"(?:is\s+fine\s*,?\s+)?but\s+){re.escape(term)}\b",
            lowered,
        )
        if not is_negative and re.search(rf"\b{re.escape(term)}\b", lowered):
            values.append(term)
    size = re.search(
        r"\b(?:shoe\s+)?size\s+(?:(?:under|below|up\s+to|at\s+most)\s+)?"
        r"(?:xxs|xs|s|m|l|xl|xxl|xxxl|\d+(?:\.\d+)?)\b",
        message,
        re.IGNORECASE,
    )
    if size:
        values.append(_clean(size.group(0)))
    return list(dict.fromkeys(values))


class DialogStateManager:
    """Deterministic per-session preference memory and question strategy.

    Active records, explicit negatives, superseded values, declined fields, and
    pending-question context remain separate. The resulting ``decision`` gives
    retrieval and ranking one testable view of the customer's current intent.
    """

    def __init__(self, broad_question_limit: int = 2) -> None:
        self.broad_question_limit = max(0, int(broad_question_limit))
        self._sessions: dict[str, dict] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        profile = copy.deepcopy(user_profile) if isinstance(user_profile, dict) else {}
        self._sessions[session_id] = {
            "user_profile": profile,
            "category": None,
            "records": [],
            "excluded": {},
            "negatives": {},
            "superseded": set(),
            "asked_attributes": [],
            "declined_attributes": set(),
            "broad_questions": 0,
            "pending_attribute": None,
            "history": [],
            "current_turn": 0,
            "last_input": None,
            "last_decision": None,
        }

    def process_turn(self, session_id: str, user_message: str, turn: int) -> dict:
        """Update one session and return the current structured search intent."""

        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before process_turn")
        try:
            normalized_turn = int(turn)
        except (TypeError, ValueError) as error:
            raise ValueError("turn must be an integer from 1 to 10") from error
        if not 1 <= normalized_turn <= 10:
            raise ValueError("turn must be an integer from 1 to 10")

        state = self._sessions[session_id]
        message = _clean(user_message, limit=2000)
        if normalized_turn == state["current_turn"]:
            if message == state["last_input"]:
                return copy.deepcopy(state["last_decision"])
            raise ValueError("a turn cannot be processed twice with different messages")
        if normalized_turn < state["current_turn"]:
            raise ValueError("turns must be processed in increasing order")

        # A short answer such as "blue" belongs to the attribute asked on the
        # preceding turn; an intent override cancels that pending association.
        pending = state["pending_attribute"]
        state["pending_attribute"] = None
        is_override = bool(OVERRIDE_SIGNAL_RE.search(message))
        if is_override and pending:
            self._cancel_interrupted_question(state, pending)
            pending = None

        state["history"].append({"turn": normalized_turn, "user_message": message})

        # Interpretation precedence matters: a decline must not become a real
        # constraint, and an override must replace rather than append intent.
        declined = self._declined_attribute(message, pending)
        if declined:
            declined_attribute, clear_existing = declined
            state["declined_attributes"].add(declined_attribute)
            if clear_existing:
                self._remove_attribute(state, declined_attribute, exclude=False)
        elif is_override:
            self._apply_override(state, message, normalized_turn)
        else:
            self._apply_information(state, message, normalized_turn, pending)

        state["current_turn"] = normalized_turn
        question, ask_attribute = self._next_question(state, normalized_turn)
        decision = self._decision(
            state,
            question,
            ask_attribute,
            is_override,
            declined[0] if declined else None,
        )
        state["last_input"] = message
        state["last_decision"] = copy.deepcopy(decision)
        return copy.deepcopy(decision)

    # ``update`` is a concise alias for the integration lead.
    update = process_turn

    def get_state(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before get_state")
        state = self._sessions[session_id]
        return {
            "user_profile": copy.deepcopy(state["user_profile"]),
            "category": state["category"],
            "active_constraints": self._active_constraints(state),
            "excluded_constraints": self._excluded_constraints(state),
            "negative_constraints": self._negative_constraints(state),
            "asked_attributes": list(state["asked_attributes"]),
            "declined_attributes": sorted(state["declined_attributes"]),
            "pending_attribute": state["pending_attribute"],
            "search_query": self._search_query(state),
            "history": copy.deepcopy(state["history"]),
            "current_turn": state["current_turn"],
        }

    def retarget_question(self, session_id: str, attribute: str) -> str:
        """Replace the just-planned question without losing answer context.

        The orchestrator may use this after retrieval to select an attribute
        that separates the current candidates.  Keeping the pending attribute
        inside the dialog manager ensures the customer's next reply is still
        stored under the correct field.
        """

        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before retarget_question")
        if attribute not in ALLOWED_ATTRIBUTES:
            raise ValueError("attribute must be one of the allowed Agent fields")
        state = self._sessions[session_id]
        if state["current_turn"] >= 10 or not isinstance(state.get("last_decision"), dict):
            raise RuntimeError("there is no replaceable pending question")
        if attribute in state["declined_attributes"]:
            raise ValueError("a declined attribute cannot be asked again")
        previous = state["pending_attribute"]
        message = QUESTION_TEXT[attribute]
        if previous == attribute:
            return message
        if previous and state["asked_attributes"] and state["asked_attributes"][-1] == previous:
            state["asked_attributes"].pop()
            if previous == "other" and state["broad_questions"]:
                state["broad_questions"] -= 1
        self._record_question(state, attribute, message)
        if isinstance(state.get("last_decision"), dict):
            state["last_decision"]["message"] = message
            state["last_decision"]["ask_attribute"] = attribute
        return message

    def _apply_information(
        self,
        state: dict,
        message: str,
        turn: int,
        pending: str | None,
    ) -> None:
        category_match = CATEGORY_RE.search(message)
        if category_match:
            self._set_category(state, category_match.group(1))

        negative_matches = list(NEGATIVE_VALUE_RE.finditer(message))
        negative_values = [
            value
            for match in negative_matches
            for value in _split_negative_values(match.group(1))
        ]
        if not GENERIC_REJECTION_RE.search(message):
            for value in negative_values:
                attribute = classify_attribute(value)
                if _looks_like_category(value):
                    attribute = "category"
                self._remove_matching_value(state, attribute, value)
                self._exclude_value(state, attribute, value, explicit_negative=True)

        requirement = REQUIREMENT_RE.search(message)
        matters = MATTERS_RE.search(message)
        if requirement:
            for value in _split_values(requirement.group(1)):
                negatives = _negative_payloads(value)
                if negatives:
                    for negative in negatives:
                        attribute = classify_attribute(negative)
                        self._remove_matching_value(state, attribute, negative)
                        self._exclude_value(
                            state, attribute, negative, explicit_negative=True
                        )
                else:
                    self._add_value(state, value, "initial_requirement", turn)
            return
        if matters:
            for value in _split_values(matters.group(1)):
                self._add_answer_value(state, value, pending, "confirmed", turn)
            return

        if negative_values:
            positive_values = _positive_residuals(message, negative_matches)
            for value in positive_values:
                if pending == "category" and _looks_like_category(value):
                    self._set_category(state, value)
                    continue
                self._add_answer_value(
                    state, value, pending, "confirmed" if pending else "direct", turn
                )
            return

        if turn == 1 and category_match:
            tail = _clean(message[category_match.end():])
            if tail.lower().startswith("but i'm still exploring") or tail.lower().startswith(
                "but i am still exploring"
            ):
                return
            if (
                tail
                and not REQUIREMENT_RE.search(tail)
                and _negative_payload(tail) is None
            ):
                self._add_value(state, tail, "initial_preference", turn)
            return

        if pending and message and not GENERIC_REJECTION_RE.search(message):
            for value in _positive_answer_values(message):
                self._add_answer_value(state, value, pending, "confirmed", turn)
            return

        if not GENERIC_REJECTION_RE.search(message):
            for value in _obvious_constraints(message):
                self._add_value(state, value, "direct", turn)

    def _apply_override(self, state: dict, message: str, turn: int) -> None:
        structured = OVERRIDE_RE.search(message)
        replacements: list[str]
        old_values: list[str]
        if structured:
            replacements = _split_values(structured.group(1))
            old_values = []
        else:
            replacements, old_values = _direct_override(message)

        lowered = message.lower()
        global_override = (
            ("ignore" in lowered and "earlier" in lowered)
            or "changed my mind" in lowered
            or "change my mind" in lowered
        )
        if global_override:
            retained: list[dict] = []
            for record in state["records"]:
                if record["source"] == "initial_preference":
                    self._exclude_value(state, record["attribute"], record["value"])
                else:
                    retained.append(record)
            state["records"] = retained

        for old_value in old_values:
            attribute = classify_attribute(old_value)
            if _looks_like_category(old_value):
                attribute = "category"
            self._remove_matching_value(state, attribute, old_value)
            self._exclude_value(state, attribute, old_value)

        for value in replacements:
            if not value:
                continue
            negatives = _negative_payloads(value)
            if negatives:
                for negative in negatives:
                    attribute = classify_attribute(negative)
                    self._remove_matching_value(state, attribute, negative)
                    self._exclude_value(
                        state, attribute, negative, explicit_negative=True
                    )
                continue
            if not structured and _looks_like_category(value):
                self._set_category(state, value, replacing=True)
                continue
            attribute = classify_attribute(value)
            if not global_override:
                self._remove_attribute(state, attribute, exclude=True)
            self._add_value(state, value, "override", turn, attribute=attribute)

    def _add_answer_value(
        self,
        state: dict,
        value: str,
        pending: str | None,
        source: str,
        turn: int,
    ) -> None:
        negatives = _negative_payloads(value)
        if negatives:
            for negative in negatives:
                detected = classify_attribute(negative)
                attribute = detected
                if pending and pending != "other" and detected == "feature":
                    attribute = pending
                self._remove_matching_value(state, attribute, negative)
                self._exclude_value(state, attribute, negative, explicit_negative=True)
            return
        if pending == "category":
            self._set_category(state, value)
            return
        detected = classify_attribute(value)
        attribute = detected
        if pending and pending != "other" and detected == "feature":
            attribute = pending
        self._add_value(state, value, source, turn, attribute=attribute)

    def _add_value(
        self,
        state: dict,
        value: str,
        source: str,
        turn: int,
        *,
        attribute: str | None = None,
    ) -> None:
        cleaned = _clean(value)
        normalized = _key(cleaned)
        if not normalized:
            return
        if source == "override":
            state["superseded"].discard(normalized)
            for excluded_values in state["excluded"].values():
                excluded_values[:] = [
                    excluded for excluded in excluded_values if _key(excluded) != normalized
                ]
            for negative_values in state["negatives"].values():
                negative_values[:] = [
                    negative for negative in negative_values if _key(negative) != normalized
                ]
        elif normalized in state["superseded"]:
            return
        if any(
            normalized == _key(excluded)
            for values in state["excluded"].values()
            for excluded in values
        ):
            return
        resolved_attribute = attribute or classify_attribute(cleaned)
        if resolved_attribute not in ALLOWED_ATTRIBUTES or resolved_attribute in {"category", "other"}:
            resolved_attribute = "feature"
        for record in state["records"]:
            if record["attribute"] == resolved_attribute and _key(record["value"]) == normalized:
                return
        state["records"].append({
            "attribute": resolved_attribute,
            "value": cleaned,
            "source": source,
            "turn": turn,
        })

    def _set_category(self, state: dict, value: str, replacing: bool = False) -> None:
        cleaned = _clean(value)
        if not cleaned:
            return
        if replacing and state["category"] and _key(state["category"]) != _key(cleaned):
            self._exclude_value(state, "category", state["category"])
        normalized = _key(cleaned)
        state["superseded"].discard(normalized)
        category_exclusions = state["excluded"].get("category", [])
        category_exclusions[:] = [
            excluded for excluded in category_exclusions if _key(excluded) != normalized
        ]
        category_negatives = state["negatives"].get("category", [])
        category_negatives[:] = [
            negative for negative in category_negatives if _key(negative) != normalized
        ]
        state["category"] = cleaned
        state["declined_attributes"].discard("category")

    def _remove_attribute(self, state: dict, attribute: str, *, exclude: bool) -> None:
        retained: list[dict] = []
        for record in state["records"]:
            if record["attribute"] == attribute:
                if exclude:
                    self._exclude_value(state, attribute, record["value"])
            else:
                retained.append(record)
        state["records"] = retained
        if attribute == "category" and state["category"]:
            if exclude:
                self._exclude_value(state, "category", state["category"])
            state["category"] = None

    def _remove_matching_value(self, state: dict, attribute: str, value: str) -> None:
        normalized = _key(value)
        state["records"] = [
            record for record in state["records"]
            if not (
                record["attribute"] == attribute
                and _key(record["value"]) in {
                    normalized,
                    f"{attribute.replace('_', ' ')} {normalized}".strip(),
                }
            )
        ]
        if attribute == "category" and state["category"] and _key(state["category"]) == normalized:
            state["category"] = None

    def _exclude_value(
        self,
        state: dict,
        attribute: str,
        value: str,
        *,
        explicit_negative: bool = False,
    ) -> None:
        cleaned = _clean(value)
        normalized = _key(cleaned)
        if not normalized:
            return
        state["superseded"].add(normalized)
        bucket = state["excluded"].setdefault(attribute, [])
        if all(_key(existing) != normalized for existing in bucket):
            bucket.append(cleaned)
        if explicit_negative:
            negative_bucket = state["negatives"].setdefault(attribute, [])
            if all(_key(existing) != normalized for existing in negative_bucket):
                negative_bucket.append(cleaned)

    def _declined_attribute(
        self,
        message: str,
        pending: str | None,
    ) -> tuple[str, bool] | None:
        normalized_message = (
            message.replace("\u2019", "'").replace("\u2018", "'").replace("\u02bc", "'")
        )
        # A qualified answer still contains useful preference information.  For
        # example, "any color except red" must not erase the color constraint.
        if DECLINE_EXCEPTION_RE.search(normalized_message):
            return None

        decline_message = normalized_message
        qualified = re.search(
            r"^(.+?),\s*but\s+(.+?)\s*[.!?]?\s*$",
            normalized_message,
            re.IGNORECASE,
        )
        if qualified:
            tail = _clean(qualified.group(2))
            benign_tail = re.fullmatch(
                r"(?:please\s+)?(?:(?:show|give|list|recommend)\s+(?:me\s+)?"
                r"(?:some\s+)?(?:options|choices|recommendations)|surprise\s+me|"
                r"use\s+your\s+judgment|go\s+ahead|thanks|thank\s+you|"
                r"that(?:'s|\s+is)\s+all)(?:\s+please)?",
                tail,
                re.IGNORECASE,
            )
            if not benign_tail:
                return None
            decline_message = _clean(qualified.group(1))

        additional = NO_ADDITIONAL_PREFERENCE_RE.search(decline_message)
        if additional:
            attribute = _decline_target(additional.group(1), pending)
            return (attribute, False) if attribute else None
        match = NO_PREFERENCE_RE.search(decline_message)
        if match:
            attribute = _decline_target(match.group(1), pending)
            return (attribute, True) if attribute else None

        for pattern in (
            DONT_MIND_RE,
            ANY_ATTRIBUTE_FINE_RE,
            ANY_ATTRIBUTE_WORKS_RE,
            OPEN_TO_ANY_ATTRIBUTE_RE,
            NOT_PICKY_RE,
            ATTRIBUTE_UP_TO_YOU_RE,
            ATTRIBUTE_DOES_NOT_MATTER_RE,
            FLEXIBLE_ATTRIBUTE_RE,
            WHATEVER_ATTRIBUTE_RE,
        ):
            paraphrase = pattern.search(decline_message)
            if paraphrase:
                attribute = _decline_target(paraphrase.group(1), pending)
                return (attribute, True) if attribute else None

        if GENERIC_DECLINE_RE.search(decline_message):
            return (pending, True) if pending else None
        if "use your judgment" in decline_message.lower():
            return (pending, True) if pending else None
        return None

    def _cancel_interrupted_question(self, state: dict, attribute: str) -> None:
        if state["asked_attributes"] and state["asked_attributes"][-1] == attribute:
            state["asked_attributes"].pop()
        if attribute == "other" and state["broad_questions"]:
            state["broad_questions"] -= 1

    def _next_question(self, state: dict, turn: int) -> tuple[str, str | None]:
        if turn >= 10:
            state["pending_attribute"] = None
            return "Based on everything you've told me, here are my top recommendations.", None

        # When a customer supplies multiple details for the requested field,
        # one bounded follow-up can reveal the rest of that same preference
        # cluster.  Mixed answers, declines, broad ``other`` questions, and
        # already-repeated attributes continue through the normal strategy.
        if state["asked_attributes"]:
            prior_attribute = state["asked_attributes"][-1]
            fresh_confirmed = [
                record
                for record in state["records"]
                if record["turn"] == turn and record["source"] == "confirmed"
            ]
            if (
                prior_attribute not in {"other", "category"}
                and len(fresh_confirmed) >= 2
                and all(record["attribute"] == prior_attribute for record in fresh_confirmed)
                and state["asked_attributes"].count(prior_attribute) < 2
            ):
                label = prior_attribute.replace("_", " ")
                return self._record_question(
                    state,
                    prior_attribute,
                    f"Is there one more {label} detail I should prioritize?",
                )

        if not state["category"] and "category" not in state["declined_attributes"]:
            return self._record_question(state, "category", QUESTION_TEXT["category"])

        if (
            state["broad_questions"] < self.broad_question_limit
            and "other" not in state["declined_attributes"]
        ):
            if state["broad_questions"] == 0:
                message = (
                    "What matters most to you—such as material, color, fit, budget, "
                    "or intended use?"
                )
            else:
                message = "Is there one more must-have detail I should prioritize?"
            state["broad_questions"] += 1
            return self._record_question(state, "other", message)

        active = self._active_constraints(state)
        for attribute in QUESTION_ORDER:
            if (
                attribute not in state["declined_attributes"]
                and attribute not in state["asked_attributes"]
                and attribute not in active
            ):
                return self._record_question(state, attribute, QUESTION_TEXT[attribute])

        return self._record_question(state, "other", QUESTION_TEXT["other"])

    @staticmethod
    def _record_question(state: dict, attribute: str, message: str) -> tuple[str, str]:
        state["asked_attributes"].append(attribute)
        state["pending_attribute"] = attribute
        return message, attribute

    def _active_constraints(self, state: dict) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        if state["category"]:
            result["category"] = [state["category"]]
        for record in state["records"]:
            result.setdefault(record["attribute"], []).append(record["value"])
        return copy.deepcopy(result)

    @staticmethod
    def _excluded_constraints(state: dict) -> dict[str, list[str]]:
        return {
            attribute: list(values)
            for attribute, values in sorted(state["excluded"].items())
            if values
        }

    @staticmethod
    def _constraint_priorities(state: dict) -> dict[str, dict[str, str]]:
        priorities: dict[str, dict[str, str]] = {}
        for record in state["records"]:
            priority = (
                "hard"
                if record["source"] in {"initial_requirement", "override"}
                else "soft"
            )
            priorities.setdefault(record["attribute"], {})[record["value"]] = priority
        return copy.deepcopy(priorities)

    @staticmethod
    def _negative_constraints(state: dict) -> dict[str, list[str]]:
        return {
            attribute: list(values)
            for attribute, values in sorted(state["negatives"].items())
            if values
        }

    def _search_query(self, state: dict) -> str:
        values: list[str] = []
        if state["category"]:
            values.append(state["category"])
        for record in state["records"]:
            if record["attribute"] != "budget":
                values.append(record["value"])
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = _key(value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(value)
        return " ".join(result)

    def _decision(
        self,
        state: dict,
        message: str,
        ask_attribute: str | None,
        is_override: bool,
        declined_attribute: str | None = None,
    ) -> dict:
        active = self._active_constraints(state)
        non_category = sum(len(values) for key, values in active.items() if key != "category")
        # This dictionary is the explicit boundary between language/state
        # handling and retrieval/ranking. Keeping it structured makes behavior
        # testable without an LLM or access to hidden labels.
        return {
            "search_query": self._search_query(state),
            "category": state["category"],
            "active_constraints": active,
            "constraint_priorities": self._constraint_priorities(state),
            "excluded_constraints": self._excluded_constraints(state),
            "negative_constraints": self._negative_constraints(state),
            "message": message,
            "ask_attribute": ask_attribute,
            "is_vague": not state["category"] or non_category == 0,
            "is_override": is_override,
            "declined_attribute": declined_attribute,
        }


# Shorter name for callers that prefer ``DialogManager``.
DialogManager = DialogStateManager
