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
    details: dict | None = None,
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
            "details": details or {},
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

    def test_known_exclusion_is_omitted_when_an_alternative_can_fill_the_limit(self) -> None:
        leather = candidate("LEATHER", "Popular leather jacket", retrieval_score=1.0)
        cotton = candidate("COTTON", "Cotton jacket", retrieval_score=0.1)

        ranked = rank_products(
            [leather, cotton],
            "jacket",
            {"category": ["jackets"]},
            {},
            top_k=1,
            excluded_constraints={"material": ["leather"]},
        )

        self.assertEqual([item["parent_asin"] for item in ranked], ["COTTON"])

    def test_known_exclusion_remains_a_last_resort_fallback(self) -> None:
        leather = candidate("LEATHER", "Popular leather jacket", retrieval_score=1.0)
        cotton = candidate("COTTON", "Cotton jacket", retrieval_score=0.1)

        ranked = rank_products(
            [leather, cotton],
            "jacket",
            {"category": ["jackets"]},
            {},
            top_k=2,
            excluded_constraints={"material": ["material: leather"]},
        )

        self.assertEqual(
            [item["parent_asin"] for item in ranked],
            ["COTTON", "LEATHER"],
        )

    def test_missing_metadata_is_not_assumed_to_violate_an_exclusion(self) -> None:
        leather = candidate("LEATHER", "Leather jacket", retrieval_score=1.0)
        incomplete = candidate("INCOMPLETE", retrieval_score=0.1)

        ranked = rank_products(
            [leather, incomplete],
            "jacket",
            {},
            {},
            top_k=1,
            excluded_constraints={"material": ["leather"]},
        )

        self.assertEqual(ranked[0]["parent_asin"], "INCOMPLETE")

    def test_negated_catalog_mention_is_not_treated_as_a_violation(self) -> None:
        compliant = candidate(
            "COMPLIANT",
            "Easy watch tool",
            retrieval_score=1.0,
            features=["Size your watch band effectively without wiggling"],
        )
        alternative = candidate("ALTERNATIVE", "Basic watch tool", retrieval_score=0.1)

        ranked = rank_products(
            [compliant, alternative],
            "easy watch tool",
            {},
            {},
            top_k=1,
            excluded_constraints={"feature": ["wiggling"]},
        )

        self.assertEqual(ranked[0]["parent_asin"], "COMPLIANT")

    def test_unnegated_mention_still_violates_an_exclusion(self) -> None:
        leather = candidate(
            "LEATHER",
            "Leather jacket",
            retrieval_score=1.0,
            features=["Do not machine wash"],
        )
        cotton = candidate("COTTON", "Cotton jacket", retrieval_score=0.1)

        ranked = rank_products(
            [leather, cotton],
            "jacket",
            {},
            {},
            top_k=1,
            excluded_constraints={"material": ["leather"]},
        )

        self.assertEqual(ranked[0]["parent_asin"], "COTTON")

    def test_exclusion_matches_qualified_noncontiguous_catalog_wording(self) -> None:
        leather = candidate(
            "LEATHER",
            "Genuine cowhide leather jacket",
            retrieval_score=1.0,
        )
        cotton = candidate("COTTON", "Cotton jacket", retrieval_score=0.1)

        ranked = rank_products(
            [leather, cotton],
            "jacket",
            {},
            {},
            top_k=1,
            excluded_constraints={"material": ["genuine leather"]},
        )

        self.assertEqual(ranked[0]["parent_asin"], "COTTON")

    def test_hard_constraint_outranks_multiple_soft_matches(self) -> None:
        hard_match = candidate(
            "HARD",
            "Technical jacket",
            retrieval_score=0.5,
            features=["waterproof shell"],
        )
        soft_match = candidate(
            "SOFT",
            "Casual comfort jacket",
            retrieval_score=0.5,
            features=["casual styling", "soft comfort"],
        )

        ranked = rank_products(
            [soft_match, hard_match],
            "jacket",
            {"feature": ["waterproof", "casual", "comfort"]},
            {},
            constraint_priorities={
                "feature": {
                    "waterproof": "hard",
                    "casual": "soft",
                    "comfort": "soft",
                }
            },
        )

        self.assertEqual(ranked[0]["parent_asin"], "HARD")

    def test_negated_catalog_feature_is_not_a_positive_match(self) -> None:
        negated = candidate(
            "NEGATED",
            "Trail shoe",
            retrieval_score=0.5,
            features=["This shoe is not waterproof during water immersion"],
        )
        positive = candidate(
            "POSITIVE",
            "Trail shoe",
            retrieval_score=0.5,
            features=["Waterproof membrane for wet trails"],
        )

        ranked = rank_products(
            [negated, positive],
            "waterproof trail shoe",
            {"feature": ["waterproof"]},
            {},
            constraint_priorities={"feature": {"waterproof": "hard"}},
        )

        self.assertEqual(ranked[0]["parent_asin"], "POSITIVE")

    def test_negation_does_not_cross_catalog_value_boundaries(self) -> None:
        cotton = candidate(
            "COTTON",
            "Everyday shirt",
            retrieval_score=0.5,
            details={"Care Instructions": "No bleach", "Material": "Cotton"},
        )
        waterproof = candidate(
            "WATERPROOF",
            "Trail jacket",
            retrieval_score=0.5,
            features=["Use without worry", "Waterproof shell"],
        )
        not_only = candidate(
            "NOT-ONLY",
            "Trail jacket",
            retrieval_score=0.5,
            features=["Not only waterproof, but breathable"],
        )

        self.assertEqual(
            rank_products(
                [candidate("OTHER", "Everyday shirt", retrieval_score=0.5), cotton],
                "cotton shirt",
                {"material": ["cotton"]},
                {},
            )[0]["parent_asin"],
            "COTTON",
        )
        for item in (waterproof, not_only):
            with self.subTest(parent_asin=item["parent_asin"]):
                ranked = rank_products(
                    [candidate("PLAIN", "Trail jacket", retrieval_score=0.5), item],
                    "waterproof trail jacket",
                    {"feature": ["waterproof"]},
                    {},
                )
                self.assertEqual(ranked[0]["parent_asin"], item["parent_asin"])

    def test_negation_does_not_cross_nested_or_punctuation_boundaries(self) -> None:
        nested_cotton = candidate(
            "Z-NESTED-COTTON",
            "Everyday shirt",
            details={
                "Specifications": {
                    "Attributes": ["No bleach", {"Material": "Cotton"}],
                }
            },
        )
        plain_shirt = candidate("A-PLAIN-SHIRT", "Everyday shirt")

        ranked = rank_products(
            [plain_shirt, nested_cotton],
            "everyday shirt",
            {"material": ["cotton"]},
            {},
            top_k=2,
        )
        self.assertEqual(ranked[0]["parent_asin"], "Z-NESTED-COTTON")

        for punctuation in (";", ".", "!"):
            with self.subTest(punctuation=punctuation):
                waterproof = candidate(
                    "Z-WATERPROOF",
                    "Trail jacket",
                    features=[f"Use without worry{punctuation} Waterproof shell"],
                )
                plain_jacket = candidate("A-PLAIN-JACKET", "Trail jacket")
                ranked = rank_products(
                    [plain_jacket, waterproof],
                    "trail jacket",
                    {"feature": ["waterproof"]},
                    {},
                    top_k=2,
                )
                self.assertEqual(ranked[0]["parent_asin"], "Z-WATERPROOF")

    def test_contractions_and_free_of_are_negative_catalog_evidence(self) -> None:
        negated = candidate(
            "A-NEGATED",
            "Trail jacket",
            features=["This jacket isn't waterproof."],
        )
        positive = candidate(
            "Z-POSITIVE",
            "Trail jacket",
            features=["Waterproof shell."],
        )

        ranked = rank_products(
            [negated, positive],
            "trail jacket",
            {"feature": ["waterproof"]},
            {},
            top_k=2,
        )
        self.assertEqual(ranked[0]["parent_asin"], "Z-POSITIVE")

        leather_free = candidate(
            "LEATHER-FREE",
            "Popular trail jacket",
            retrieval_score=1.0,
            features=["Free of leather and animal-derived materials."],
        )
        incomplete = candidate(
            "INCOMPLETE",
            "Basic trail jacket",
            retrieval_score=0.1,
        )
        ranked = rank_products(
            [leather_free, incomplete],
            "trail jacket",
            {},
            {},
            top_k=1,
            excluded_constraints={"material": ["leather"]},
        )
        self.assertEqual(ranked[0]["parent_asin"], "LEATHER-FREE")

    def test_unmatched_hard_constraint_does_not_remove_incomplete_candidates(self) -> None:
        partial = candidate(
            "PARTIAL",
            "Waterproof jacket",
            retrieval_score=0.1,
        )
        otherwise_relevant = candidate(
            "RELEVANT",
            "Everyday jacket",
            retrieval_score=1.0,
        )

        ranked = rank_products(
            [partial, otherwise_relevant],
            "jacket",
            {"feature": ["waterproof membrane"]},
            {},
            constraint_priorities={"feature": "hard"},
        )

        self.assertEqual(ranked[0]["parent_asin"], "RELEVANT")
        self.assertEqual(len(ranked), 2)

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

    def test_bare_budget_answer_uses_structured_budget_context(self) -> None:
        within = candidate("WITHIN", "Winter jacket", retrieval_score=0.1, price=45.0)
        over = candidate("OVER", "Winter jacket", retrieval_score=1.0, price=80.0)

        ranked = rank_products(
            [over, within],
            "winter jacket",
            {"budget": ["$50"]},
            {},
            top_k=2,
        )

        self.assertEqual(ranked[0]["parent_asin"], "WITHIN")

    def test_around_budget_is_a_soft_price_proximity_signal(self) -> None:
        near = candidate("NEAR", "Winter jacket", retrieval_score=0.50, price=55.0)
        far = candidate("FAR", "Winter jacket", retrieval_score=0.51, price=150.0)
        unknown = candidate("UNKNOWN", "Winter jacket", retrieval_score=0.49, price=None)

        ranked = rank_products(
            [far, unknown, near],
            "winter jacket",
            {"budget": ["budget around $50"]},
            {},
            top_k=3,
        )

        self.assertEqual(ranked[0]["parent_asin"], "NEAR")
        self.assertEqual({item["parent_asin"] for item in ranked}, {"NEAR", "FAR", "UNKNOWN"})

    def test_budget_range_prefers_known_prices_inside_range(self) -> None:
        within = candidate("WITHIN", "Walking shoe", retrieval_score=0.1, price=60.0)
        below = candidate("BELOW", "Walking shoe", retrieval_score=1.0, price=20.0)

        ranked = rank_products(
            [below, within],
            "walking shoe",
            {"budget": ["between $40 and $70"]},
            {},
            top_k=2,
        )

        self.assertEqual(ranked[0]["parent_asin"], "WITHIN")

    def test_compact_budget_ranges_are_supported_with_safe_context(self) -> None:
        within = candidate("WITHIN", "Walking shoe", retrieval_score=0.1, price=75.0)
        below = candidate("BELOW", "Walking shoe", retrieval_score=1.0, price=20.0)

        for message, constraints in (
            ("walking shoe for $50-$100", {}),
            ("walking shoe", {"budget": ["50 to 100"]}),
            ("walking shoe", {"budget": ["I can spend $50-$100"]}),
        ):
            with self.subTest(message=message, constraints=constraints):
                ranked = rank_products(
                    [below, within],
                    message,
                    constraints,
                    {},
                    top_k=2,
                )
                self.assertEqual(ranked[0]["parent_asin"], "WITHIN")

    def test_budget_parsing_supports_comma_amounts_and_unprefixed_ranges(self) -> None:
        within_maximum = candidate(
            "WITHIN-MAXIMUM",
            "Technical walking shoe",
            retrieval_score=1.0,
            price=900.0,
        )
        cheap = candidate(
            "CHEAP",
            "Generic shoe",
            retrieval_score=0.1,
            price=1.0,
        )
        ranked = rank_products(
            [cheap, within_maximum],
            "technical walking shoe under $1,000",
            {"budget": ["under $1"]},
            {},
            top_k=2,
        )
        self.assertEqual(ranked[0]["parent_asin"], "WITHIN-MAXIMUM")

        within_range = candidate(
            "WITHIN-RANGE",
            "Walking shoe",
            retrieval_score=0.1,
            price=1500.0,
        )
        below_range = candidate(
            "BELOW-RANGE",
            "Walking shoe",
            retrieval_score=1.0,
            price=100.0,
        )
        for message, constraints in (
            ("walking shoe", {"budget": ["$1,000-$2,000"]}),
            ("walking shoe with a budget between 1,000 and 2,000", {}),
        ):
            with self.subTest(message=message, constraints=constraints):
                ranked = rank_products(
                    [below_range, within_range],
                    message,
                    constraints,
                    {},
                    top_k=2,
                )
                self.assertEqual(ranked[0]["parent_asin"], "WITHIN-RANGE")

    def test_multiple_minimum_budgets_use_the_strongest_lower_bound(self) -> None:
        below_strongest = candidate(
            "BELOW-STRONGEST",
            "Walking shoe",
            retrieval_score=1.0,
            price=150.0,
        )
        within_strongest = candidate(
            "WITHIN-STRONGEST",
            "Walking shoe",
            retrieval_score=0.1,
            price=250.0,
        )

        ranked = rank_products(
            [below_strongest, within_strongest],
            "walking shoe",
            {"budget": ["at least $100", "at least $200"]},
            {},
            top_k=2,
        )

        self.assertEqual(ranked[0]["parent_asin"], "WITHIN-STRONGEST")

    def test_size_under_number_is_not_interpreted_as_a_budget(self) -> None:
        relevant = candidate(
            "RELEVANT",
            "Walking shoe size under 10",
            retrieval_score=1.0,
            price=100.0,
        )
        cheap = candidate(
            "CHEAP",
            "Generic walking shoe",
            retrieval_score=0.1,
            price=5.0,
        )

        for constraints in (
            {},
            {"size": ["under 10"]},
            {"size": ["under 10"], "budget": ["under 10"]},
        ):
            with self.subTest(constraints=constraints):
                ranked = rank_products(
                    [cheap, relevant],
                    "walking shoe size under 10",
                    constraints,
                    {},
                    top_k=2,
                )
                self.assertEqual(ranked[0]["parent_asin"], "RELEVANT")

    def test_free_form_numeric_range_is_not_assumed_to_be_a_budget(self) -> None:
        relevant = candidate("RELEVANT", "Size 8 to 10 walking shoe", retrieval_score=1.0, price=150.0)
        cheap = candidate("CHEAP", "Generic walking shoe", retrieval_score=0.1, price=9.0)

        ranked = rank_products(
            [cheap, relevant],
            "walking shoe size 8 to 10",
            {"size": ["8 to 10"]},
            {},
            top_k=2,
        )

        self.assertEqual(ranked[0]["parent_asin"], "RELEVANT")

        for phrase in ("size from 8 to 10", "sizes between 4 and 8"):
            with self.subTest(phrase=phrase):
                ranked = rank_products(
                    [cheap, relevant],
                    f"walking shoe {phrase}",
                    {"size": [phrase]},
                    {},
                    top_k=2,
                )
                self.assertEqual(ranked[0]["parent_asin"], "RELEVANT")

    def test_single_character_and_decimal_sizes_affect_ranking(self) -> None:
        size_s = candidate(
            "SIZE-S",
            "Classic shirt",
            retrieval_score=0.5,
            details={"Size": "S"},
        )
        size_m = candidate(
            "SIZE-M",
            "Classic shirt",
            retrieval_score=0.5,
            details={"Size": "M"},
        )
        size_8 = candidate(
            "SIZE-8",
            "Walking shoe",
            retrieval_score=0.5,
            details={"Size": "8"},
        )
        size_85 = candidate(
            "SIZE-85",
            "Walking shoe",
            retrieval_score=0.5,
            details={"Size": "8.5"},
        )

        shirts = rank_products(
            [size_m, size_s],
            "classic shirt",
            {"size": ["S"]},
            {},
            top_k=2,
        )
        shoes = rank_products(
            [size_8, size_85],
            "walking shoe",
            {"size": ["8.5"]},
            {},
            top_k=2,
        )

        self.assertEqual(shirts[0]["parent_asin"], "SIZE-S")
        self.assertEqual(shoes[0]["parent_asin"], "SIZE-85")

    def test_known_single_character_size_exclusion_is_respected(self) -> None:
        size_s = candidate(
            "SIZE-S",
            "Classic shirt",
            retrieval_score=1.0,
            details={"Size": "S"},
        )
        size_m = candidate(
            "SIZE-M",
            "Classic shirt",
            retrieval_score=0.1,
            details={"Size": "M"},
        )

        ranked = rank_products(
            [size_s, size_m],
            "classic shirt",
            {},
            {},
            top_k=1,
            excluded_constraints={"size": ["S"]},
        )

        self.assertEqual(ranked[0]["parent_asin"], "SIZE-M")

    def test_height_measurements_do_not_count_as_numeric_size_evidence(self) -> None:
        model_height = candidate(
            "A-MODEL-HEIGHT",
            "Casual dress",
            features=[
                "Models wear size S, 5'8.5\" tall. Caralyn wears size XL.",
                "Model size is 5'8.5\" tall.",
            ],
            details={"Model Size": "5'8.5\""},
        )
        actual_size = candidate(
            "Z-SIZE-85",
            "Casual dress",
            details={"Size": "8.5"},
        )
        actual_size_5 = candidate(
            "Z-SIZE-5",
            "Casual dress",
            details={"Size": "5"},
        )

        decimal_ranked = rank_products(
            [model_height, actual_size],
            "casual dress",
            {"size": ["8.5"]},
            {},
            top_k=2,
        )
        whole_ranked = rank_products(
            [model_height, actual_size_5],
            "casual dress",
            {"size": ["5"]},
            {},
            top_k=2,
        )

        self.assertEqual(decimal_ranked[0]["parent_asin"], "Z-SIZE-85")
        self.assertEqual(whole_ranked[0]["parent_asin"], "Z-SIZE-5")

    def test_height_measurement_does_not_trigger_a_size_exclusion(self) -> None:
        model_height = candidate(
            "MODEL-HEIGHT",
            "Casual dress",
            retrieval_score=1.0,
            features=["Model size is 5'8.5\" tall and wears XL."],
        )
        alternative = candidate(
            "ALTERNATIVE",
            "Casual dress",
            retrieval_score=0.1,
        )

        ranked = rank_products(
            [model_height, alternative],
            "casual dress",
            {},
            {},
            top_k=1,
            excluded_constraints={"size": ["8.5"]},
        )

        self.assertEqual(ranked[0]["parent_asin"], "MODEL-HEIGHT")

    def test_structured_physical_dimensions_do_not_become_sizes(self) -> None:
        for dimension, requested_size in (
            ("6.5X3.5 Inches", "6.5"),
            ('3" x 3"', "3"),
            ("5 x 8", "8"),
            ("19 x 13 x 8-Inch", "8"),
        ):
            with self.subTest(dimension=dimension):
                misleading = candidate(
                    "A-DIMENSION",
                    "Walking shoe",
                    details={"Size": dimension},
                )
                true_size = candidate(
                    "Z-TRUE-SIZE",
                    "Walking shoe",
                    details={"Size": requested_size},
                )
                ranked = rank_products(
                    [misleading, true_size],
                    f"walking shoe size {requested_size}",
                    {"size": [requested_size]},
                    {},
                    top_k=2,
                )
                self.assertEqual(ranked[0]["parent_asin"], "Z-TRUE-SIZE")

    def test_immediate_explicit_free_text_size_list_is_recognized(self) -> None:
        listed = candidate(
            "Z-LISTED",
            "Walking shoe",
            features=["Available sizes: 8, 8.5, 9."],
        )
        not_listed = candidate(
            "A-NOT-LISTED",
            "Walking shoe",
            features=["Available sizes: 8 and 9."],
        )

        ranked = rank_products(
            [not_listed, listed],
            "walking shoe",
            {"size": ["8.5"]},
            {},
            top_k=2,
        )

        self.assertEqual(ranked[0]["parent_asin"], "Z-LISTED")

    def test_immediate_size_chart_list_is_recognized(self) -> None:
        chart = candidate(
            "Z-CHART",
            "Classic fitted tunic",
            features=["Size chart: S / M / L"],
        )
        plain = candidate("A-PLAIN", "Classic fitted tunic")

        for requested_size in ("S", "M", "L"):
            with self.subTest(requested_size=requested_size):
                ranked = rank_products(
                    [plain, chart],
                    "classic fitted tunic",
                    {"size": [requested_size]},
                    {},
                    top_k=2,
                )
                self.assertEqual(ranked[0]["parent_asin"], "Z-CHART")

    def test_product_measurements_are_not_treated_as_a_budget(self) -> None:
        exact = candidate(
            "EXACT",
            "Gold chronograph watch",
            retrieval_score=1.0,
            price=80.0,
            features=["Band fits up to 8-inch wrists", "Up to 100-hour chronograph"],
        )
        cheap = candidate(
            "CHEAP",
            "Basic watch",
            retrieval_score=0.1,
            price=5.0,
        )

        ranked = rank_products(
            [cheap, exact],
            "gold chronograph watch with a band up to 8-inch wrists and up to 100-hour timing",
            {"feature": ["band fits up to 8-inch wrists"]},
            {},
            top_k=2,
        )

        self.assertEqual(ranked[0]["parent_asin"], "EXACT")

    def test_real_budget_still_applies_alongside_measurements(self) -> None:
        within = candidate("WITHIN", "Basic watch", retrieval_score=0.1, price=40.0)
        over = candidate(
            "OVER",
            "Gold chronograph watch",
            retrieval_score=1.0,
            price=80.0,
            features=["Band fits up to 8-inch wrists"],
        )

        ranked = rank_products(
            [over, within],
            "gold watch with a band up to 8-inch wrists under $50",
            {"feature": ["band fits up to 8-inch wrists"], "budget": ["under $50"]},
            {},
            top_k=2,
        )

        self.assertEqual(ranked[0]["parent_asin"], "WITHIN")

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
