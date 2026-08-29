from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from starter.retrieval import CatalogRetriever, _expanded_terms, _route_scores


PRODUCTS = [
    {
        "parent_asin": "RUN-BLUE",
        "title": "Blue trail running shoe",
        "features": ["Lightweight mesh", "Outdoor grip"],
        "description": ["A sneaker for running on trails"],
        "price": 49.0,
        "categories": ["Women", "Shoes", "Running"],
        "details": {"Department": "womens"},
        "average_rating": 4.5,
        "rating_number": 100,
        "store": "Trail Brand",
    },
    {
        "parent_asin": "RUN-RED",
        "title": "Red road running sneakers",
        "features": ["Cushioned sole"],
        "description": ["A lightweight road shoe"],
        "price": 60.0,
        "categories": ["Men", "Shoes", "Running"],
        "details": {"Department": "mens"},
        "average_rating": 4.2,
        "rating_number": 80,
        "store": "Road Brand",
    },
    {
        "parent_asin": "BOOT-LEATHER",
        "title": "Brown winter boot",
        "features": ["Genuine leather", "Warm lining"],
        "description": ["Outdoor footwear for winter"],
        "price": 90.0,
        "categories": ["Women", "Shoes", "Boots"],
        "details": {"Material": "Leather"},
        "average_rating": 4.7,
        "rating_number": 300,
        "store": "Winter Brand",
    },
    {
        "parent_asin": "SHIRT-COMFORT",
        "title": "Everyday cotton shirt",
        "features": ["Comfort fit", "Soft fabric"],
        "description": ["Casual short sleeve top"],
        "price": 25.0,
        "categories": ["Men", "Clothing", "Shirts"],
        "details": {},
        "average_rating": 4.8,
        "rating_number": 500,
        "store": "Basics",
    },
    {
        "parent_asin": "BAG-BLACK",
        "title": "Black leather handbag",
        "features": ["Shoulder strap"],
        "description": ["A purse for daily use"],
        "price": None,
        "categories": ["Women", "Handbags"],
        "details": {},
        "average_rating": 4.0,
        "rating_number": 20,
        "store": "Bag Store",
    },
    {
        "parent_asin": "EMPTY",
        "title": "Simple accessory",
        "features": [],
        "description": [],
        "price": None,
        "categories": [],
        "details": {},
        "average_rating": 0.0,
        "rating_number": 0,
        "store": "",
    },
]


class RetrievalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.catalog_path = Path(cls.temporary_directory.name) / "catalog.jsonl"
        cls.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in PRODUCTS),
            encoding="utf-8",
        )
        cls.retriever = CatalogRetriever(cls.catalog_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.retriever.close()
        cls.temporary_directory.cleanup()

    def test_returns_ranker_candidate_contract(self) -> None:
        results = self.retriever.retrieve_products("blue running shoe", top_k=3)

        self.assertTrue(results)
        candidate = results[0]
        self.assertEqual(candidate["parent_asin"], "RUN-BLUE")
        self.assertEqual(candidate["product"]["price"], 49.0)
        self.assertIn("current_message", candidate["route_hits"])
        self.assertEqual(candidate["retrieval_score"], 1.0)

    def test_title_match_is_ranked_first_for_current_message(self) -> None:
        results = self.retriever.retrieve_products("blue trail running", top_k=5)

        self.assertEqual(results[0]["parent_asin"], "RUN-BLUE")

    def test_plural_and_synonym_expansion_improves_recall(self) -> None:
        results = self.retriever.retrieve_products("sneakers", top_k=6)
        identifiers = {candidate["parent_asin"] for candidate in results}

        self.assertIn("RUN-BLUE", identifiers)
        self.assertIn("RUN-RED", identifiers)
        self.assertIn("dresses", _expanded_terms("dress"))
        self.assertIn("dress", _expanded_terms("dresses"))

    def test_multiple_routes_merge_without_duplicate_products(self) -> None:
        results = self.retriever.retrieve_products(
            "running shoes",
            active_constraints={"category": ["running shoes"], "color": ["blue"]},
            category="Shoes",
            top_k=6,
        )
        by_asin = {candidate["parent_asin"]: candidate for candidate in results}

        self.assertEqual(len(results), len(by_asin))
        self.assertEqual(
            by_asin["RUN-BLUE"]["route_hits"],
            ["current_message", "active_constraints", "category"],
        )

    def test_current_message_route_outweighs_profile_only_route(self) -> None:
        results = self.retriever.retrieve_products(
            "winter boot",
            user_profile={"preference_tags": ["comfort", "shirt"]},
            top_k=6,
        )

        self.assertEqual(results[0]["parent_asin"], "BOOT-LEATHER")
        shirt = next(item for item in results if item["parent_asin"] == "SHIRT-COMFORT")
        self.assertEqual(shirt["route_hits"], ["profile"])

    def test_active_constraint_can_retrieve_a_matching_product(self) -> None:
        results = self.retriever.retrieve_products(
            "something for winter",
            active_constraints={"material": ["leather"]},
            top_k=6,
        )
        identifiers = {candidate["parent_asin"] for candidate in results}

        self.assertIn("BOOT-LEATHER", identifiers)
        self.assertIn("active_constraints", next(
            item["route_hits"] for item in results if item["parent_asin"] == "BOOT-LEATHER"
        ))

    def test_empty_query_and_invalid_limit_are_safe(self) -> None:
        self.assertEqual(self.retriever.retrieve_products("the and please"), [])
        self.assertEqual(self.retriever.retrieve_products("running", top_k=0), [])
        self.assertEqual(self.retriever.retrieve_products("running", top_k="bad"), [])

    def test_missing_catalog_fields_do_not_crash(self) -> None:
        results = self.retriever.retrieve_products("simple accessory", top_k=6)

        self.assertEqual(results[0]["parent_asin"], "EMPTY")

    def test_inputs_are_not_mutated(self) -> None:
        constraints = {"color": ["blue"], "category": ["running"]}
        profile = {"preference_tags": ["comfort"]}
        original_constraints = copy.deepcopy(constraints)
        original_profile = copy.deepcopy(profile)

        self.retriever.retrieve_products(
            "running shoes",
            active_constraints=constraints,
            user_profile=profile,
        )

        self.assertEqual(constraints, original_constraints)
        self.assertEqual(profile, original_profile)

    def test_output_is_deterministic_and_respects_top_k(self) -> None:
        first = self.retriever.retrieve_products("shoes", top_k=2)
        second = self.retriever.retrieve_products("shoes", top_k=2)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

    def test_raw_bm25_scores_are_inverted_and_normalized(self) -> None:
        normalized = _route_scores([-5.0, -3.0, -1.0])

        self.assertEqual(normalized[0], 1.0)
        self.assertEqual(normalized[-1], 0.0)
        self.assertGreater(normalized[0], normalized[1])


if __name__ == "__main__":
    unittest.main()
