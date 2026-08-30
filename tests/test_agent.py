from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from starter.agent import (
    CATALOG_PATH_ENV,
    EARLY_RECOMMENDATION_LIMIT,
    Agent,
    MAX_RECOMMENDATIONS,
    _candidate_question_attribute,
    _fallback_rank,
    _resolve_catalog_path,
)
from starter.dialog import ALLOWED_ATTRIBUTES


PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.5,
    "rating_style": "usually positive",
    "preference_tags": ["comfort", "durability"],
    "summary": "Prior purchases emphasize comfort and durability.",
}


def product(
    parent_asin: str,
    title: str,
    *,
    features: list[str],
    price: float = 30.0,
    rating: float = 4.5,
    ratings: int = 100,
) -> dict:
    return {
        "parent_asin": parent_asin,
        "title": title,
        "features": features,
        "description": ["Comfortable everyday clothing"],
        "price": price,
        "categories": ["Clothing", "Shirts"],
        "details": {},
        "average_rating": rating,
        "rating_number": ratings,
        "store": "Test Store",
    }


class AgentIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.catalog_path = Path(cls.temporary_directory.name) / "catalog.jsonl"
        cls.products = [
            *[
                product(
                    f"BLUE-{index:02}",
                    f"Blue cotton running shirt {index}",
                    features=["breathable cotton", "blue color", "gym running"],
                    price=20.0 + index,
                    rating=4.8,
                    ratings=500 - index,
                )
                for index in range(15)
            ],
            *[
                product(
                    f"RED-{index:02}",
                    f"Red polyester casual shirt {index}",
                    features=["polyester fabric", "red color", "casual style"],
                    price=25.0 + index,
                    rating=4.2,
                    ratings=200 - index,
                )
                for index in range(15)
            ],
        ]
        cls.catalog_path.write_text(
            "".join(json.dumps(item) + "\n" for item in cls.products),
            encoding="utf-8",
        )
        cls.by_asin = {item["parent_asin"]: item for item in cls.products}
        cls.agent = Agent(cls.catalog_path, candidate_pool_size=100)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.agent.close()
        cls.temporary_directory.cleanup()

    def session_id(self, suffix: str = "") -> str:
        return f"{self._testMethodName}-{suffix}"

    def reset(self, suffix: str = "", profile: dict | None = None) -> str:
        session_id = self.session_id(suffix)
        self.agent.reset(session_id, PROFILE if profile is None else profile)
        return session_id

    def test_reset_is_required(self) -> None:
        with self.assertRaises(RuntimeError):
            self.agent.respond("missing", "shirts", 1, 10)

    def test_default_catalog_path_does_not_depend_on_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {}, clear=True):
                prior = Path.cwd()
                try:
                    os.chdir(directory)
                    resolved = _resolve_catalog_path(None)
                finally:
                    os.chdir(prior)

        self.assertTrue(resolved.is_file())
        self.assertEqual(resolved.name, "catalog.jsonl")

    def test_catalog_environment_override_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "custom.jsonl"
            catalog.write_text("", encoding="utf-8")
            with mock.patch.dict(os.environ, {CATALOG_PATH_ENV: str(catalog)}):
                self.assertEqual(_resolve_catalog_path(None), catalog)

    def test_response_matches_exact_agent_contract(self) -> None:
        session_id = self.reset()
        response = self.agent.respond(
            session_id,
            "I'm looking for shirts. A key requirement is: blue cotton.",
            1,
            5,
        )

        self.assertEqual(
            set(response),
            {"message", "ask_attribute", "recommendations", "usage"},
        )
        self.assertIsInstance(response["message"], str)
        self.assertIn(response["ask_attribute"], ALLOWED_ATTRIBUTES | {None})
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})
        self.assertLessEqual(len(response["recommendations"]), 5)
        self.assertTrue(response["recommendations"])
        self.assertTrue(all(
            set(recommendation) == {"parent_asin"}
            for recommendation in response["recommendations"]
        ))

    def test_early_turns_return_a_short_focused_list(self) -> None:
        session_id = self.reset()
        response = self.agent.respond(
            session_id, "I'm looking for shirts, but I'm still exploring.", 1, 100
        )

        self.assertEqual(len(response["recommendations"]), EARLY_RECOMMENDATION_LIMIT)

    def test_later_turns_expand_to_competition_maximum(self) -> None:
        session_id = self.reset()
        response = self.agent.respond(
            session_id, "I'm looking for shirts, but I'm still exploring.", 4, 100
        )

        self.assertEqual(len(response["recommendations"]), MAX_RECOMMENDATIONS)

    def test_constraints_flow_through_retrieval_and_ranking(self) -> None:
        session_id = self.reset()
        response = self.agent.respond(
            session_id,
            "I'm looking for shirts. A key requirement is: cotton; color: blue.",
            1,
            10,
        )

        selected = [item["parent_asin"] for item in response["recommendations"]]
        self.assertTrue(selected)
        self.assertTrue(all("blue" in self.by_asin[asin]["title"].lower() for asin in selected))

    def test_candidate_question_uses_balanced_catalog_evidence(self) -> None:
        candidates = [
            {
                "parent_asin": f"C-{index}",
                "product": product(
                    f"C-{index}",
                    f"{'Blue' if index < 4 else 'Red'} cotton shirt",
                    features=["cotton fabric"],
                ),
            }
            for index in range(8)
        ]

        self.assertEqual(
            _candidate_question_attribute(candidates, set(), set()),
            "color",
        )
        self.assertIsNone(
            _candidate_question_attribute(candidates, {"color", "material"}, set())
        )

    def test_decline_retargets_question_and_preserves_pending_context(self) -> None:
        session_id = self.reset()
        self.agent.respond(
            session_id, "I'm looking for shirts, but I'm still exploring.", 1, 10
        )
        response = self.agent.respond(
            session_id,
            "I don't have a preference for other; please use your judgment.",
            2,
            10,
        )

        self.assertIn(response["ask_attribute"], {"color", "material"})
        state = self.agent.dialog.get_state(session_id)
        self.assertEqual(state["pending_attribute"], response["ask_attribute"])

        self.agent.respond(session_id, "Blue.", 3, 10)
        state = self.agent.dialog.get_state(session_id)
        self.assertEqual(state["active_constraints"]["color"], ["Blue"])

    def test_exclusions_and_priority_metadata_flow_to_ranking(self) -> None:
        session_id = self.reset()
        decision = {
            "search_query": "shirts blue",
            "category": "shirts",
            "active_constraints": {"color": ["blue"]},
            "excluded_constraints": {"style": ["casual"]},
            "negative_constraints": {"material": ["polyester"]},
            "constraint_priorities": {"color": {"blue": "hard"}},
            "message": "Do you have another preference?",
            "ask_attribute": "other",
            "is_override": False,
        }

        with (
            mock.patch.object(self.agent.dialog, "process_turn", return_value=decision),
            mock.patch.object(self.agent, "_retrieve_candidates", return_value=[]),
            mock.patch("starter.agent.rank_products", return_value=[]) as rank,
        ):
            self.agent.respond(session_id, "blue shirts, not polyester", 1, 10)

        self.assertEqual(
            rank.call_args.kwargs["excluded_constraints"],
            {"material": ["polyester"]},
        )
        self.assertEqual(
            rank.call_args.kwargs["constraint_priorities"],
            {"color": {"blue": "hard"}},
        )

    def test_continuing_session_explores_unseen_products(self) -> None:
        session_id = self.reset()
        first = self.agent.respond(
            session_id, "I'm looking for shirts, but I'm still exploring.", 1, 10
        )
        second = self.agent.respond(
            session_id, "I don't have an additional preference for other.", 2, 10
        )

        first_asins = {item["parent_asin"] for item in first["recommendations"]}
        second_asins = {item["parent_asin"] for item in second["recommendations"]}
        self.assertEqual(len(first_asins), EARLY_RECOMMENDATION_LIMIT)
        self.assertEqual(len(second_asins), EARLY_RECOMMENDATION_LIMIT)
        self.assertTrue(first_asins.isdisjoint(second_asins))

    def test_override_makes_pre_override_products_eligible_again(self) -> None:
        session_id = self.reset()
        first = self.agent.respond(
            session_id, "I'm looking for shirts. color: red.", 1, 10
        )
        overridden = self.agent.respond(
            session_id,
            "Actually, ignore my earlier preference. What I need is: color: red.",
            2,
            10,
        )

        self.assertEqual(first["recommendations"], overridden["recommendations"])

    def test_switch_override_resets_seen_products_for_the_new_intent(self) -> None:
        session_id = self.reset()
        red_first = self.agent.respond(
            session_id, "I'm looking for shirts. color: red.", 1, 10
        )
        blue = self.agent.respond(
            session_id, "Switch from red to blue.", 2, 10
        )
        red_again = self.agent.respond(
            session_id, "Switch from blue to red.", 3, 10
        )

        self.assertNotEqual(red_first["recommendations"], blue["recommendations"])
        self.assertEqual(red_first["recommendations"], red_again["recommendations"])

    def test_duplicate_request_is_cached_and_defensively_copied(self) -> None:
        session_id = self.reset()
        message = "I'm looking for blue shirts, but I'm still exploring."
        first = self.agent.respond(session_id, message, 1, 10)
        expected = copy.deepcopy(first)
        first["recommendations"].clear()
        repeated = self.agent.respond(session_id, message, 1, 10)

        self.assertEqual(repeated, expected)
        self.assertEqual(
            len(self.agent._sessions[session_id]["seen_asins"]),
            EARLY_RECOMMENDATION_LIMIT,
        )

    def test_reset_clears_seen_products_and_cached_response(self) -> None:
        session_id = self.reset()
        message = "I'm looking for shirts, but I'm still exploring."
        first = self.agent.respond(session_id, message, 1, 10)
        self.agent.reset(session_id, PROFILE)
        after_reset = self.agent.respond(session_id, message, 1, 10)

        self.assertEqual(first, after_reset)

    def test_sessions_are_isolated(self) -> None:
        first_session = self.reset("first")
        second_session = self.reset("second")
        message = "I'm looking for shirts, but I'm still exploring."

        first = self.agent.respond(first_session, message, 1, 10)
        second = self.agent.respond(second_session, message, 1, 10)

        self.assertEqual(first, second)

    def test_profile_is_defensively_copied(self) -> None:
        profile = copy.deepcopy(PROFILE)
        session_id = self.reset(profile=profile)
        profile["preference_tags"].append("mutated")

        self.assertNotIn(
            "mutated",
            self.agent._sessions[session_id]["user_profile"]["preference_tags"],
        )

    def test_retrieval_failure_still_returns_a_valid_question(self) -> None:
        session_id = self.reset()
        self.agent._candidate_cache.clear()
        with mock.patch.object(
            self.agent.retriever,
            "retrieve_products",
            side_effect=RuntimeError("retrieval failed"),
        ):
            response = self.agent.respond(
                session_id, "I'm looking for shirts, but I'm still exploring.", 1, 10
            )

        self.assertIsInstance(response["message"], str)
        self.assertIn(response["ask_attribute"], ALLOWED_ATTRIBUTES)
        self.assertEqual(response["recommendations"], [])

    def test_unchanged_search_state_reuses_bounded_candidate_cache(self) -> None:
        session_id = self.reset()
        candidates = [{"parent_asin": "A", "retrieval_score": 1.0}]
        with mock.patch.object(
            self.agent.retriever,
            "retrieve_products",
            return_value=candidates,
        ) as retrieve:
            self.agent.respond(
                session_id, "I'm looking for belts, but I'm still exploring.", 1, 10
            )
            self.agent.respond(
                session_id, "I don't have an additional preference for other.", 2, 10
            )

        self.assertEqual(retrieve.call_count, 1)

    def test_cache_distinguishes_different_fallback_queries(self) -> None:
        first_session = self.reset("fallback-one")
        second_session = self.reset("fallback-two")
        self.agent._candidate_cache.clear()
        with mock.patch.object(
            self.agent.retriever,
            "retrieve_products",
            return_value=[],
        ) as retrieve:
            self.agent.respond(first_session, "unparsed alpha request", 1, 10)
            self.agent.respond(second_session, "unparsed beta request", 1, 10)

        self.assertEqual(retrieve.call_count, 2)
        self.assertEqual(retrieve.call_args_list[0].args[0], "unparsed alpha request")
        self.assertEqual(retrieve.call_args_list[1].args[0], "unparsed beta request")

    def test_ranking_failure_uses_deterministic_retrieval_fallback(self) -> None:
        session_id = self.reset()
        self.agent._candidate_cache.clear()
        candidates = [
            {"parent_asin": "LOW", "retrieval_score": 0.1},
            {"parent_asin": "HIGH", "retrieval_score": 0.9},
        ]
        with (
            mock.patch.object(
                self.agent.retriever,
                "retrieve_products",
                return_value=candidates,
            ),
            mock.patch.object(
                self.agent.retriever,
                "retrieve_strict_products",
                return_value=[],
            ),
            mock.patch("starter.agent.rank_products", side_effect=RuntimeError("rank failed")),
        ):
            response = self.agent.respond(
                session_id, "I'm looking for shirts, but I'm still exploring.", 1, 10
            )

        self.assertEqual(
            response["recommendations"],
            [{"parent_asin": "HIGH"}, {"parent_asin": "LOW"}],
        )

    def test_selection_deduplicates_and_never_repeats_seen_products(self) -> None:
        selected = Agent._select_recommendations(
            [
                {"parent_asin": "SEEN"},
                {"parent_asin": "NEW"},
                {"parent_asin": "NEW"},
            ],
            {"SEEN"},
            2,
        )

        self.assertEqual(
            selected,
            [{"parent_asin": "NEW"}],
        )

    def test_selection_uses_popularity_only_inside_the_relevance_batch(self) -> None:
        selected = Agent._select_recommendations(
            [
                {"parent_asin": "FIRST", "product": {"rating_number": 0}},
                {"parent_asin": "POPULAR", "product": {"rating_number": 1000}},
                {"parent_asin": "OUTSIDE", "product": {"rating_number": 1_000_000}},
            ],
            set(),
            2,
        )

        self.assertEqual(
            selected,
            [{"parent_asin": "POPULAR"}, {"parent_asin": "FIRST"}],
        )
        self.assertNotIn({"parent_asin": "OUTSIDE"}, selected)

    def test_popularity_respects_the_maximum_relevance_promotion(self) -> None:
        for original_index in range(10):
            with self.subTest(original_index=original_index):
                candidates = [
                    {
                        "parent_asin": f"ITEM-{index}",
                        "product": {
                            "rating_number": 1_000_000 if index == original_index else 0
                        },
                    }
                    for index in range(10)
                ]
                selected = Agent._select_recommendations(candidates, set(), 10)
                order = [item["parent_asin"] for item in selected]

                self.assertGreaterEqual(
                    order.index(f"ITEM-{original_index}"),
                    max(0, original_index - 4),
                )

    def test_selection_tolerates_invalid_popularity_values(self) -> None:
        selected = Agent._select_recommendations(
            [
                {"parent_asin": "A", "product": {"rating_number": "unknown"}},
                {"parent_asin": "B", "product": {"rating_number": float("nan")}},
                {"parent_asin": "C"},
            ],
            set(),
            3,
        )

        self.assertEqual(
            selected,
            [
                {"parent_asin": "A"},
                {"parent_asin": "B"},
                {"parent_asin": "C"},
            ],
        )

    def test_turn_ten_returns_no_follow_up_question(self) -> None:
        session_id = self.reset()
        response = self.agent.respond(
            session_id, "I'm looking for blue shirts.", 10, 10
        )

        self.assertIsNone(response["ask_attribute"])
        self.assertIsInstance(response["message"], str)

    def test_invalid_top_k_returns_no_recommendations(self) -> None:
        session_id = self.reset()
        response = self.agent.respond(
            session_id, "I'm looking for shirts, but I'm still exploring.", 1, "bad"
        )

        self.assertEqual(response["recommendations"], [])

    def test_fallback_rank_ignores_malformed_candidates(self) -> None:
        ranked = _fallback_rank([
            None,
            {},
            {"parent_asin": "B", "retrieval_score": float("nan")},
            {"parent_asin": "A", "retrieval_score": 0.5},
        ])

        self.assertEqual([item["parent_asin"] for item in ranked], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
