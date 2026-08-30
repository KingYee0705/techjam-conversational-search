from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from starter.retrieval import (
    CatalogRetriever,
    STRICT_SCORE_FLOOR,
    _constraint_fts_expression,
    _expanded_terms,
    _fts_expression,
    _product_size_terms,
    _route_scores,
    _size_search_term,
    _strict_fts_expression,
)


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
    {
        "parent_asin": "SIZE-LETTERS",
        "title": "Classic fitted shirt",
        "features": ["Available in S M L"],
        "description": [],
        "price": 20.0,
        "categories": ["Men", "Clothing", "Shirts"],
        "details": {"Size": "S M L"},
        "average_rating": 4.0,
        "rating_number": 10,
        "store": "Sizing Test",
    },
    {
        "parent_asin": "SIZE-8",
        "title": "Numeric size walking shoe",
        "features": ["US size 8"],
        "description": [],
        "price": 30.0,
        "categories": ["Men", "Shoes", "Walking"],
        "details": {"Size": "8"},
        "average_rating": 4.0,
        "rating_number": 10,
        "store": "Sizing Test",
    },
    {
        "parent_asin": "SIZE-85",
        "title": "Half size walking shoe",
        "features": ["US size 8.5"],
        "description": [],
        "price": 30.0,
        "categories": ["Men", "Shoes", "Walking"],
        "details": {"Size": "8.5"},
        "average_rating": 4.0,
        "rating_number": 10,
        "store": "Sizing Test",
    },
    {
        "parent_asin": "FALSE-SIZE-85",
        "title": "Walking shoe model information",
        "features": ["Model wears size S and is 5'8.5\" tall"],
        "description": [],
        "price": 30.0,
        "categories": ["Men", "Shoes", "Walking"],
        "details": {"Model Size": "5'8.5\""},
        "average_rating": 4.0,
        "rating_number": 10,
        "store": "Sizing Test",
    },
    {
        "parent_asin": "SIZE-CHART",
        "title": "Chart-only fitted tunic",
        "features": ["Size chart: S / M / L"],
        "description": [],
        "price": 24.0,
        "categories": ["Women", "Clothing", "Tunics"],
        "details": {},
        "average_rating": 4.0,
        "rating_number": 10,
        "store": "Sizing Test",
    },
    {
        "parent_asin": "DIMENSION-2D",
        "title": "Flat picture frame",
        "features": [],
        "description": [],
        "price": 12.0,
        "categories": ["Home", "Frames"],
        "details": {"Size": "5 x 8"},
        "average_rating": 4.0,
        "rating_number": 10,
        "store": "Dimension Test",
    },
    {
        "parent_asin": "DIMENSION-3D",
        "title": "Rectangular storage bin",
        "features": [],
        "description": [],
        "price": 18.0,
        "categories": ["Home", "Storage"],
        "details": {"Size": "19 x 13 x 8-Inch"},
        "average_rating": 4.0,
        "rating_number": 10,
        "store": "Dimension Test",
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

    def test_strict_expression_ands_concepts_and_strips_labels(self) -> None:
        expression = _strict_fts_expression(
            "Running Shoes",
            {"color": ["color: blue"], "budget": ["under $10"]},
        )

        self.assertIn(" AND ", expression)
        self.assertIn('"blue"', expression)
        self.assertIn('"shoe"', expression)
        self.assertIn(" OR ", expression)
        self.assertNotIn('"color"', expression)
        self.assertNotIn('"10"', expression)

    def test_size_constraints_preserve_structured_short_and_numeric_values(self) -> None:
        constraints = {"size": ["S", "M", "L", "8", "8.5"]}
        broad = _constraint_fts_expression(constraints)
        strict = _strict_fts_expression(None, constraints)

        for value in ("s", "m", "l", "8", "8.5"):
            term = _size_search_term(value)
            self.assertIn(f'"{term}"', broad)
            self.assertIn(f'("{term}")', strict)

        # The same one-character tokens remain excluded without structured
        # size context, so ordinary prose queries do not become noisier.
        self.assertEqual(_fts_expression("S M L 8 8.5"), "")

    def test_strict_size_route_finds_letter_integer_and_decimal_sizes(self) -> None:
        for value in ("S", "M", "L"):
            results = self.retriever.retrieve_strict_products(
                category="Shirts",
                active_constraints={"size": [value]},
                top_k=20,
            )
            self.assertIn(
                "SIZE-LETTERS",
                {candidate["parent_asin"] for candidate in results},
            )

        size_eight = self.retriever.retrieve_strict_products(
            category="Walking Shoes",
            active_constraints={"size": ["8"]},
            top_k=20,
        )
        size_eight_half = self.retriever.retrieve_strict_products(
            category="Walking Shoes",
            active_constraints={"size": ["8.5"]},
            top_k=20,
        )

        self.assertIn("SIZE-8", {item["parent_asin"] for item in size_eight})
        self.assertEqual(
            [item["parent_asin"] for item in size_eight_half],
            ["SIZE-85"],
        )

    def test_dimensions_do_not_emit_or_index_synthetic_sizes(self) -> None:
        dimension_products = [
            product for product in PRODUCTS
            if product["parent_asin"].startswith("DIMENSION-")
        ]
        for product in dimension_products:
            self.assertEqual(_product_size_terms(product), [])

        size_eight = self.retriever.retrieve_strict_products(
            active_constraints={"size": ["8"]},
            top_k=20,
        )
        identifiers = {item["parent_asin"] for item in size_eight}
        self.assertIn("SIZE-8", identifiers)
        self.assertNotIn("DIMENSION-2D", identifiers)
        self.assertNotIn("DIMENSION-3D", identifiers)

    def test_immediate_size_chart_context_is_synthetically_indexed(self) -> None:
        chart_product = next(
            product for product in PRODUCTS
            if product["parent_asin"] == "SIZE-CHART"
        )
        self.assertEqual(
            _product_size_terms(chart_product),
            [_size_search_term(value) for value in ("S", "M", "L")],
        )

        for value in ("S", "M", "L"):
            results = self.retriever.retrieve_strict_products(
                category="Tunics",
                active_constraints={"size": [value]},
                top_k=20,
            )
            self.assertEqual(
                [item["parent_asin"] for item in results],
                ["SIZE-CHART"],
            )

    def test_strict_route_requires_every_disclosed_concept(self) -> None:
        results = self.retriever.retrieve_strict_products(
            category="Running Shoes",
            active_constraints={"color": ["color: blue"], "feature": ["mesh"]},
            top_k=6,
        )

        self.assertEqual([item["parent_asin"] for item in results], ["RUN-BLUE"])
        self.assertEqual(
            results[0]["route_hits"],
            ["category", "active_constraints"],
        )
        self.assertGreaterEqual(results[0]["retrieval_score"], STRICT_SCORE_FLOOR)

    def test_strict_category_only_route_reports_only_category_evidence(self) -> None:
        results = self.retriever.retrieve_strict_products(
            category="Shoes",
            top_k=2,
        )

        self.assertTrue(results)
        self.assertTrue(all(item["route_hits"] == ["category"] for item in results))

        duplicated_category = self.retriever.retrieve_strict_products(
            category="Shoes",
            active_constraints={"category": ["Shoes"]},
            top_k=2,
        )
        self.assertTrue(duplicated_category)
        self.assertTrue(
            all(item["route_hits"] == ["category"] for item in duplicated_category)
        )

    def test_strict_route_is_bounded_and_safe_when_empty(self) -> None:
        first = self.retriever.retrieve_strict_products(category="Shoes", top_k=1)
        second = self.retriever.retrieve_strict_products(category="Shoes", top_k=1)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertEqual(self.retriever.retrieve_strict_products(top_k=6), [])
        self.assertEqual(
            self.retriever.retrieve_strict_products(category="Shoes", top_k=0),
            [],
        )

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
        first = self.retriever.retrieve_products(
            "shoes",
            active_constraints={"color": ["blue"]},
            category="Shoes",
            top_k=2,
        )
        second = self.retriever.retrieve_products(
            "shoes",
            active_constraints={"color": ["blue"]},
            category="Shoes",
            top_k=2,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

    def test_raw_bm25_scores_are_inverted_and_normalized(self) -> None:
        normalized = _route_scores([-5.0, -3.0, -1.0])

        self.assertEqual(normalized[0], 1.0)
        self.assertEqual(normalized[-1], 0.0)
        self.assertGreater(normalized[0], normalized[1])


if __name__ == "__main__":
    unittest.main()
