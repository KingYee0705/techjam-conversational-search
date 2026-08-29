from __future__ import annotations

import copy
import unittest

from starter.ranking import rank_products


def candidate(
    parent_asin: str,
    title: str = "",
    *,
    retrieval_score: float = 0.5,
    price: float | None = None,
    features: list[str] | None = None,
    categories: list[str] | None = None,
    average_rating: float = 0.0,
    rating_number: int = 0,
    route_hits: list[str] | None = None,
) -> dict:
    return {
        "parent_asin": parent_asin,
        "product": {
            "title": title,
            "features": features or [],
            "description": [],
            "price": price,
            "categories": categories or [],
            "details": {},
            "average_rating": average_rating,
            "rating_number": rating_number,
            "store": "",
        },
        "retrieval_score": retrieval_score,
        "route_hits": route_hits or [],
    }


class RankingTest(unittest.TestCase):
    def test_current_message_match_beats_profile_only_match(self) -> None:
        exact = candidate("EXACT", "Blue trail running shoes")
        profile_only = candidate("PROFILE", "Comfort recovery sandals", features=["comfortable fit"])

        ranked = rank_products(
            [profile_only, exact],
            "I want blue trail running shoes",
            {},
            {"preference_tags": ["comfort", "fit"]},
        )

        self.assertEqual(ranked[0]["parent_asin"], "EXACT")

    def test_current_intent_and_constraint_outweigh_conflicting_profile(self) -> None:
        blue = candidate("BLUE", "Blue walking shoes")
        red = candidate("RED", "Red walking shoes")

        ranked = rank_products(
            [red, blue],
            "Actually, I need blue walking shoes",
            {"color": ["blue"]},
            {"preference_tags": ["red"]},
        )

        self.assertEqual(ranked[0]["parent_asin"], "BLUE")

    def test_attribute_constraint_boosts_match_without_filtering_unknowns(self) -> None:
        blue = candidate("BLUE", "Everyday shirt", features=["navy blue cotton fabric"])
        unknown = candidate("UNKNOWN", "Everyday shirt")
        red = candidate("RED", "Everyday shirt", features=["red polyester fabric"])

        ranked = rank_products(
            [unknown, red, blue],
            "everyday shirt",
            {"color": ["blue"], "material": ["cotton"]},
            {},
            top_k=3,
        )

        self.assertEqual(ranked[0]["parent_asin"], "BLUE")
        self.assertEqual({item["parent_asin"] for item in ranked}, {"BLUE", "UNKNOWN", "RED"})

    def test_budget_orders_within_unknown_then_over_budget(self) -> None:
        within = candidate("WITHIN", "Winter jacket", retrieval_score=0.1, price=45.0)
        unknown = candidate("UNKNOWN", "Winter jacket", retrieval_score=0.9, price=None)
        over = candidate("OVER", "Winter jacket", retrieval_score=1.0, price=80.0)

        ranked = rank_products(
            [over, unknown, within],
            "winter jacket under $50",
            {"budget": ["under $50"]},
            {},
            top_k=3,
        )

        self.assertEqual(
            [item["parent_asin"] for item in ranked],
            ["WITHIN", "UNKNOWN", "OVER"],
        )

    def test_budget_excludes_over_budget_when_ten_viable_exist(self) -> None:
        viable = [candidate(f"V{index:02}", "T-shirt", price=20.0) for index in range(10)]
        over = candidate("OVER", "T-shirt", retrieval_score=1.0, price=100.0)

        ranked = rank_products(
            [over, *viable],
            "T-shirt at most $30",
            {"budget": ["at most $30"]},
            {},
        )

        self.assertNotIn("OVER", [item["parent_asin"] for item in ranked])
        self.assertEqual(len(ranked), 10)

    def test_retrieval_score_wins_when_other_evidence_is_equal(self) -> None:
        low = candidate("LOW", "Black socks", retrieval_score=0.2)
        high = candidate("HIGH", "Black socks", retrieval_score=0.9)

        ranked = rank_products([low, high], "black socks", {}, {})

        self.assertEqual(ranked[0]["parent_asin"], "HIGH")

    def test_quality_is_a_small_tie_breaker(self) -> None:
        low_quality = candidate(
            "A-LOW",
            "Running shoe",
            average_rating=2.0,
            rating_number=2,
        )
        high_quality = candidate(
            "Z-HIGH",
            "Running shoe",
            average_rating=4.8,
            rating_number=1000,
        )

        ranked = rank_products([low_quality, high_quality], "running shoe", {}, {})

        self.assertEqual(ranked[0]["parent_asin"], "Z-HIGH")

    def test_deduplicates_without_mutating_inputs(self) -> None:
        first = candidate("SAME", "Walking boot", retrieval_score=0.2, route_hits=["category"])
        second = candidate("SAME", "Walking boot", retrieval_score=0.9, route_hits=["current_message"])
        inputs = [first, second]
        original = copy.deepcopy(inputs)

        ranked = rank_products(inputs, "walking boot", {}, {})

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["retrieval_score"], 0.9)
        self.assertEqual(ranked[0]["route_hits"], ["category", "current_message"])
        self.assertEqual(inputs, original)

    def test_incomplete_and_empty_candidates_are_safe(self) -> None:
        ranked = rank_products(
            [
                {},
                {"parent_asin": ""},
                {"parent_asin": "VALID", "product": {}},
                {"parent_asin": "FLAT", "title": "Simple belt"},
            ],
            "belt",
            {},
            {},
        )

        self.assertEqual({item["parent_asin"] for item in ranked}, {"VALID", "FLAT"})
        self.assertEqual(rank_products([], "anything", {}, {}), [])

    def test_output_is_deterministic_and_respects_top_k(self) -> None:
        candidates = [candidate(f"A{index}", "Plain shirt") for index in range(5, -1, -1)]

        first = rank_products(candidates, "plain shirt", {}, {}, top_k=3)
        second = rank_products(candidates, "plain shirt", {}, {}, top_k=3)

        self.assertEqual(first, second)
        self.assertEqual([item["parent_asin"] for item in first], ["A0", "A1", "A2"])


if __name__ == "__main__":
    unittest.main()
