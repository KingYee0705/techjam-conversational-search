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
    "silk", "rayon", "fabric",
)
COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown",
    "gray", "grey", "purple", "yellow", "orange",
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
    r"\b(?:i(?:'m| am)\s+)?looking\s+for\s+(.+?)(?=[.,;]|$)",
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
    r"\b(?:i\s+)?(?:don't|do\s+not)\s+have\s+(?:an?\s+)?additional\s+"
    r"preference\s+for\s+([a-z_ -]+?)(?=[;,.]|$)",
    re.IGNORECASE,
)
NO_PREFERENCE_RE = re.compile(
    r"\b(?:i\s+)?(?:don't|do\s+not)\s+have\s+(?:an?\s+)?"
    r"preference\s+for\s+([a-z_ -]+?)(?=[;,.]|$)",
    re.IGNORECASE,
)
OVERRIDE_SIGNAL_RE = re.compile(
    r"\b(?:actually|instead|ignore\s+(?:my\s+)?earlier|changed?\s+my\s+mind|"
    r"rather\s+than|make\s+it|what\s+i\s+need\s+is)\b",
    re.IGNORECASE,
)
GENERIC_REJECTION_RE = re.compile(
    r"\b(?:options\s+are\s+not\s+quite\s+right|ask\s+me\s+about\s+one\s+specific)\b",
    re.IGNORECASE,
)
NEGATIVE_VALUE_RE = re.compile(
    r"\b(?:not|without|avoid(?:ing)?)\s+([a-z][a-z0-9 -]{0,60}?)"
    r"(?=[,.;]|\s+but\b|$)",
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
    return re.sub(r"\s+", " ", str(value)).strip(" \t\n.,;")[:limit].rstrip()


def _key(value: object) -> str:
    return " ".join(token.lower() for token in TOKEN_RE.findall(_clean(value)))


def classify_attribute(value: str) -> str:
    """Classify a preference using the evaluator's public attribute rules."""

    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
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
    for attribute in ATTRIBUTE_ORDER:
        if attribute == "use_case":
            if {"use", "case"}.issubset(words):
                return attribute
        elif attribute in words:
            return attribute
    return None


def _split_values(value: str) -> list[str]:
    return [cleaned for item in value.split(";") if (cleaned := _clean(item))]


def _negative_payload(value: str) -> str | None:
    match = re.match(
        r"^(?:not|without|avoid(?:ing)?)\s+(.+?)$",
        _clean(value),
        re.IGNORECASE,
    )
    return _clean(match.group(1)) if match else None


def _looks_like_category(value: str) -> bool:
    tokens = TOKEN_RE.findall(value.lower())
    return len(tokens) <= 7 and any(token in CATEGORY_TERMS for token in tokens)


def _direct_override(message: str) -> tuple[list[str], list[str]]:
    """Return (replacement values, explicitly superseded values)."""

    instead_of = re.search(
        r"(?:actually\s*,?\s*)?(?:i\s+(?:need|want)\s+|make\s+it\s+)?"
        r"(.+?)\s+instead\s+of\s+(.+?)\s*\.?\s*$",
        message,
        re.IGNORECASE,
    )
    if instead_of:
        return [_clean(instead_of.group(1))], [_clean(instead_of.group(2))]

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

    make_it = re.search(
        r"\bmake\s+it\s+(.+?)(?:\s+instead)?\s*\.?\s*$",
        message,
        re.IGNORECASE,
    )
    if make_it:
        return [_clean(make_it.group(1))], []

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
    budget = re.search(
        r"\b(?:under|below|up\s+to|at\s+most|maximum(?:\s+of)?)\s*"
        r"(?:usd\s*)?\$?\s*\d+(?:\.\d+)?\b",
        message,
        re.IGNORECASE,
    )
    if budget:
        values.append(_clean(budget.group(0)))
    lowered = message.lower()
    for term in (*MATERIALS, *COLORS):
        is_negative = re.search(
            rf"\b(?:not|without|avoid(?:ing)?)\s+{re.escape(term)}\b",
            lowered,
        )
        if not is_negative and re.search(rf"\b{re.escape(term)}\b", lowered):
            values.append(term)
    size = re.search(
        r"\bsize\s+(?:xxs|xs|s|m|l|xl|xxl|xxxl|\d+(?:\.\d+)?)\b",
        message,
        re.IGNORECASE,
    )
    if size:
        values.append(_clean(size.group(0)))
    return list(dict.fromkeys(values))


class DialogStateManager:
    """Deterministic per-session preference memory and question strategy.

    Person 4 can pass ``search_query``, ``active_constraints``, and ``category``
    directly to ``CatalogRetriever``, then pass the same query and constraints to
    ``rank_products``.
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

        pending = state["pending_attribute"]
        state["pending_attribute"] = None
        is_override = bool(OVERRIDE_SIGNAL_RE.search(message))
        if is_override and pending:
            self._cancel_interrupted_question(state, pending)
            pending = None

        state["history"].append({"turn": normalized_turn, "user_message": message})

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
        decision = self._decision(state, question, ask_attribute, is_override)
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
            "asked_attributes": list(state["asked_attributes"]),
            "declined_attributes": sorted(state["declined_attributes"]),
            "pending_attribute": state["pending_attribute"],
            "search_query": self._search_query(state),
            "history": copy.deepcopy(state["history"]),
            "current_turn": state["current_turn"],
        }

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

        negative_values = [_clean(match.group(1)) for match in NEGATIVE_VALUE_RE.finditer(message)]
        if not GENERIC_REJECTION_RE.search(message):
            for value in negative_values:
                attribute = classify_attribute(value)
                self._remove_matching_value(state, attribute, value)
                self._exclude_value(state, attribute, value)

        requirement = REQUIREMENT_RE.search(message)
        matters = MATTERS_RE.search(message)
        if requirement:
            for value in _split_values(requirement.group(1)):
                negative = _negative_payload(value)
                if negative:
                    attribute = classify_attribute(negative)
                    self._remove_matching_value(state, attribute, negative)
                    self._exclude_value(state, attribute, negative)
                else:
                    self._add_value(state, value, "initial_requirement", turn)
            return
        if matters:
            for value in _split_values(matters.group(1)):
                self._add_answer_value(state, value, pending, "confirmed", turn)
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
                and not re.match(r"^(?:not|without|avoid(?:ing)?)\b", tail, re.IGNORECASE)
            ):
                self._add_value(state, tail, "initial_preference", turn)
            return

        if pending and message and not GENERIC_REJECTION_RE.search(message):
            self._add_answer_value(state, message, pending, "confirmed", turn)
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
            negative = _negative_payload(value)
            if negative:
                attribute = classify_attribute(negative)
                self._remove_matching_value(state, attribute, negative)
                self._exclude_value(state, attribute, negative)
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
        negative = _negative_payload(value)
        if negative:
            detected = classify_attribute(negative)
            attribute = detected
            if pending and pending != "other" and detected == "feature":
                attribute = pending
            self._remove_matching_value(state, attribute, negative)
            self._exclude_value(state, attribute, negative)
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
                record["attribute"] == attribute and _key(record["value"]) == normalized
            )
        ]
        if attribute == "category" and state["category"] and _key(state["category"]) == normalized:
            state["category"] = None

    def _exclude_value(self, state: dict, attribute: str, value: str) -> None:
        cleaned = _clean(value)
        normalized = _key(cleaned)
        if not normalized:
            return
        state["superseded"].add(normalized)
        bucket = state["excluded"].setdefault(attribute, [])
        if all(_key(existing) != normalized for existing in bucket):
            bucket.append(cleaned)

    def _declined_attribute(
        self,
        message: str,
        pending: str | None,
    ) -> tuple[str, bool] | None:
        additional = NO_ADDITIONAL_PREFERENCE_RE.search(message)
        if additional:
            attribute = _attribute_from_text(additional.group(1)) or pending
            return (attribute, False) if attribute else None
        match = NO_PREFERENCE_RE.search(message)
        if match:
            attribute = _attribute_from_text(match.group(1)) or pending
            return (attribute, True) if attribute else None
        if "use your judgment" in message.lower():
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
    ) -> dict:
        active = self._active_constraints(state)
        non_category = sum(len(values) for key, values in active.items() if key != "category")
        return {
            "search_query": self._search_query(state),
            "category": state["category"],
            "active_constraints": active,
            "excluded_constraints": self._excluded_constraints(state),
            "message": message,
            "ask_attribute": ask_attribute,
            "is_vague": not state["category"] or non_category == 0,
            "is_override": is_override,
        }


# Shorter name for callers that prefer ``DialogManager``.
DialogManager = DialogStateManager
