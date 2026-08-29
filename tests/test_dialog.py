from __future__ import annotations

import copy
import unittest

from starter.dialog import ALLOWED_ATTRIBUTES, DialogStateManager, classify_attribute


PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.5,
    "rating_style": "usually positive",
    "preference_tags": ["comfort", "durability"],
    "summary": "Prior purchases emphasize comfort and durability.",
}


class DialogStateManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = DialogStateManager()
        self.manager.reset("session", PROFILE)

    def test_reset_is_required(self) -> None:
        manager = DialogStateManager()

        with self.assertRaises(RuntimeError):
            manager.process_turn("missing", "I am looking for shoes.", 1)
        with self.assertRaises(RuntimeError):
            manager.get_state("missing")

    def test_browsing_message_extracts_category_without_exploring_text(self) -> None:
        decision = self.manager.process_turn(
            "session",
            "I'm looking for running shoes, but I'm still exploring.",
            1,
        )

        self.assertEqual(decision["category"], "running shoes")
        self.assertEqual(decision["active_constraints"], {"category": ["running shoes"]})
        self.assertEqual(decision["search_query"], "running shoes")
        self.assertEqual(decision["ask_attribute"], "other")
        self.assertTrue(decision["is_vague"])
        self.assertNotIn("exploring", decision["search_query"])

    def test_buying_message_extracts_category_and_requirement(self) -> None:
        decision = self.manager.process_turn(
            "session",
            "I'm looking for winter boots. A key requirement is: genuine leather.",
            1,
        )

        self.assertEqual(decision["category"], "winter boots")
        self.assertEqual(decision["active_constraints"]["material"], ["genuine leather"])
        self.assertIn("genuine leather", decision["search_query"])
        self.assertFalse(decision["is_vague"])

    def test_two_broad_questions_collect_multiple_constraint_types(self) -> None:
        first = self.manager.process_turn(
            "session", "I'm looking for shirts, but I'm still exploring.", 1
        )
        second = self.manager.process_turn(
            "session",
            "For that, what matters is: cotton; color: blue.",
            2,
        )

        self.assertEqual(first["ask_attribute"], "other")
        self.assertEqual(second["ask_attribute"], "other")
        self.assertEqual(second["active_constraints"]["material"], ["cotton"])
        self.assertEqual(second["active_constraints"]["color"], ["color: blue"])
        self.assertEqual(second["search_query"], "shirts cotton color: blue")

    def test_pending_category_question_interprets_short_answer(self) -> None:
        first = self.manager.process_turn("session", "I need something useful.", 1)
        second = self.manager.process_turn("session", "Trail running shoes", 2)

        self.assertEqual(first["ask_attribute"], "category")
        self.assertEqual(second["category"], "Trail running shoes")
        self.assertEqual(
            second["active_constraints"]["category"], ["Trail running shoes"]
        )

    def test_pending_specific_question_supplies_context_for_short_answer(self) -> None:
        manager = DialogStateManager(broad_question_limit=0)
        manager.reset("specific", PROFILE)
        first = manager.process_turn(
            "specific", "I'm looking for jackets, but I'm still exploring.", 1
        )
        second = manager.process_turn("specific", "Machine washable, please.", 2)

        self.assertEqual(first["ask_attribute"], "feature")
        self.assertEqual(
            second["active_constraints"]["feature"], ["Machine washable, please"]
        )

    def test_boundary_reply_is_not_stored_as_a_constraint(self) -> None:
        self.manager.process_turn(
            "session", "I'm looking for handbags, but I'm still exploring.", 1
        )
        decision = self.manager.process_turn(
            "session",
            "I don't have a preference for other; please use your judgment.",
            2,
        )
        state = self.manager.get_state("session")

        self.assertEqual(decision["active_constraints"], {"category": ["handbags"]})
        self.assertNotEqual(decision["ask_attribute"], "other")
        self.assertIn("other", state["declined_attributes"])
        self.assertNotIn("preference", decision["search_query"].lower())
        self.assertNotIn("judgment", decision["search_query"].lower())

    def test_no_additional_preference_adapts_question_strategy(self) -> None:
        self.manager.process_turn(
            "session", "I'm looking for sandals, but I'm still exploring.", 1
        )
        decision = self.manager.process_turn(
            "session", "I don't have an additional preference for other.", 2
        )

        self.assertEqual(decision["ask_attribute"], "feature")
        self.assertIn(decision["ask_attribute"], ALLOWED_ATTRIBUTES)

    def test_no_additional_preference_preserves_an_existing_constraint(self) -> None:
        self.manager.process_turn(
            "session",
            "I'm looking for shirts. A key requirement is: cotton.",
            1,
        )
        decision = self.manager.process_turn(
            "session", "I don't have an additional preference for material.", 2
        )

        self.assertEqual(decision["active_constraints"]["material"], ["cotton"])
        self.assertIn("material", self.manager.get_state("session")["declined_attributes"])

    def test_global_override_removes_only_initial_preference(self) -> None:
        self.manager.process_turn(
            "session", "I'm looking for walking shoes. extra cushioning.", 1
        )
        self.manager.process_turn(
            "session", "For that, what matters is: waterproof.", 2
        )
        decision = self.manager.process_turn(
            "session",
            "Actually, ignore my earlier preference. What I need is: leather.",
            3,
        )

        self.assertTrue(decision["is_override"])
        self.assertEqual(decision["active_constraints"]["feature"], ["waterproof"])
        self.assertEqual(decision["active_constraints"]["material"], ["leather"])
        self.assertEqual(
            decision["excluded_constraints"]["feature"], ["extra cushioning"]
        )
        self.assertNotIn("extra cushioning", decision["search_query"])
        self.assertIn("waterproof", decision["search_query"])

    def test_global_override_preserves_confirmed_constraint_of_same_attribute(self) -> None:
        self.manager.process_turn(
            "session", "I'm looking for walking shoes. extra cushioning.", 1
        )
        self.manager.process_turn(
            "session", "For that, what matters is: waterproof.", 2
        )
        decision = self.manager.process_turn(
            "session",
            "Actually, ignore my earlier preference. What I need is: zippered pockets.",
            3,
        )

        self.assertEqual(
            decision["active_constraints"]["feature"],
            ["waterproof", "zippered pockets"],
        )

    def test_duplicate_override_replacement_remains_active(self) -> None:
        self.manager.process_turn(
            "session", "I'm looking for boots. extra cushioning.", 1
        )
        self.manager.process_turn(
            "session", "For that, what matters is: leather.", 2
        )
        decision = self.manager.process_turn(
            "session",
            "Actually, ignore my earlier preference. What I need is: leather.",
            3,
        )

        self.assertEqual(decision["active_constraints"]["material"], ["leather"])
        self.assertNotIn("material", decision["excluded_constraints"])

    def test_override_replaces_a_conflicting_attribute(self) -> None:
        self.manager.process_turn(
            "session", "I'm looking for walking shoes. color: red.", 1
        )
        decision = self.manager.process_turn(
            "session", "Actually, make it blue instead.", 2
        )

        self.assertEqual(decision["active_constraints"]["color"], ["blue"])
        self.assertEqual(decision["excluded_constraints"]["color"], ["color: red"])
        self.assertNotIn("red", decision["search_query"].lower())

    def test_category_can_be_replaced_directly(self) -> None:
        self.manager.process_turn("session", "I'm looking for shoes. comfortable.", 1)
        decision = self.manager.process_turn(
            "session", "Actually, I need boots instead of shoes.", 2
        )

        self.assertEqual(decision["category"], "boots")
        self.assertEqual(decision["excluded_constraints"]["category"], ["shoes"])
        self.assertNotIn("shoes", decision["search_query"].lower())

    def test_rather_than_and_changed_mind_paraphrases_are_not_discarded(self) -> None:
        first = self.manager.process_turn(
            "session", "I'm looking for boots rather than shoes.", 1
        )
        second = self.manager.process_turn(
            "session", "I changed my mind; I want blue jackets.", 2
        )

        self.assertEqual(first["category"], "boots")
        self.assertEqual(first["excluded_constraints"]["category"], ["shoes"])
        self.assertEqual(second["category"], "blue jackets")
        self.assertNotIn("boots", second["search_query"].lower())

    def test_reintroduced_category_is_removed_from_exclusions(self) -> None:
        self.manager.process_turn("session", "I'm looking for shoes.", 1)
        self.manager.process_turn(
            "session", "Actually, I need boots instead of shoes.", 2
        )
        decision = self.manager.process_turn(
            "session", "Actually, I need shoes instead of boots.", 3
        )

        self.assertEqual(decision["category"], "shoes")
        self.assertEqual(decision["excluded_constraints"]["category"], ["boots"])

    def test_negative_preference_is_excluded_not_boosted(self) -> None:
        decision = self.manager.process_turn(
            "session", "I'm looking for jackets. not leather.", 1
        )

        self.assertNotIn("material", decision["active_constraints"])
        self.assertEqual(decision["excluded_constraints"]["material"], ["leather"])
        self.assertNotIn("leather", decision["search_query"].lower())

    def test_structured_negative_requirement_is_not_stored_as_positive(self) -> None:
        decision = self.manager.process_turn(
            "session",
            "I'm looking for jackets. A key requirement is: not leather.",
            1,
        )

        self.assertNotIn("material", decision["active_constraints"])
        self.assertEqual(decision["excluded_constraints"]["material"], ["leather"])
        self.assertEqual(decision["search_query"], "jackets")

    def test_profile_tags_are_copied_but_never_become_active_constraints(self) -> None:
        profile = copy.deepcopy(PROFILE)
        profile["preference_tags"] = ["red", "comfort"]
        manager = DialogStateManager()
        manager.reset("profile", profile)
        profile["preference_tags"].append("mutated")

        decision = manager.process_turn(
            "profile", "I'm looking for shirts. color: blue.", 1
        )
        state = manager.get_state("profile")

        self.assertEqual(decision["active_constraints"]["color"], ["color: blue"])
        self.assertNotIn("red", decision["search_query"].lower())
        self.assertNotIn("mutated", state["user_profile"]["preference_tags"])

    def test_budget_is_structured_but_omitted_from_fts_search_query(self) -> None:
        decision = self.manager.process_turn(
            "session",
            "I'm looking for jackets. A key requirement is: budget under $50.",
            1,
        )

        self.assertEqual(decision["active_constraints"]["budget"], ["budget under $50"])
        self.assertEqual(decision["search_query"], "jackets")

    def test_duplicate_turn_and_constraint_are_idempotent(self) -> None:
        first = self.manager.process_turn(
            "session",
            "I'm looking for shirts. A key requirement is: cotton.",
            1,
        )
        repeated = self.manager.process_turn(
            "session",
            "I'm looking for shirts. A key requirement is: cotton.",
            1,
        )

        self.assertEqual(first, repeated)
        self.assertEqual(repeated["active_constraints"]["material"], ["cotton"])
        self.assertEqual(len(self.manager.get_state("session")["history"]), 1)

    def test_sessions_are_independent_and_reset_clears_old_state(self) -> None:
        self.manager.reset("other-session", {})
        self.manager.process_turn("session", "I'm looking for blue shirts.", 1)
        other = self.manager.process_turn(
            "other-session", "I'm looking for leather boots.", 1
        )

        self.assertNotIn("blue shirts", other["search_query"].lower())
        self.manager.reset("session", {})
        cleared = self.manager.get_state("session")
        self.assertEqual(cleared["active_constraints"], {})
        self.assertEqual(cleared["history"], [])

    def test_returned_data_cannot_mutate_internal_state(self) -> None:
        decision = self.manager.process_turn(
            "session", "I'm looking for shirts. cotton.", 1
        )
        decision["active_constraints"]["material"].append("polyester")
        snapshot = self.manager.get_state("session")
        snapshot["active_constraints"]["material"].append("nylon")

        fresh = self.manager.get_state("session")
        self.assertEqual(fresh["active_constraints"]["material"], ["cotton"])

    def test_all_questions_are_allowed_and_turn_ten_asks_nothing(self) -> None:
        manager = DialogStateManager()
        manager.reset("questions", {})
        message = "I'm looking for accessories, but I'm still exploring."
        for turn in range(1, 10):
            decision = manager.process_turn("questions", message, turn)
            self.assertIn(decision["ask_attribute"], ALLOWED_ATTRIBUTES)
            asked = decision["ask_attribute"]
            message = f"I don't have an additional preference for {asked}."

        final = manager.process_turn("questions", message, 10)
        self.assertIsNone(final["ask_attribute"])

    def test_empty_inputs_are_safe_and_classifier_matches_contract(self) -> None:
        manager = DialogStateManager()
        manager.reset("empty", None)
        decision = manager.process_turn("empty", "", 1)

        self.assertEqual(decision["ask_attribute"], "category")
        self.assertEqual(decision["active_constraints"], {})
        self.assertEqual(classify_attribute("under $20"), "budget")
        self.assertEqual(classify_attribute("soft wool"), "material")
        self.assertEqual(classify_attribute("navy color"), "color")
        self.assertEqual(classify_attribute("wide width"), "size")
        self.assertEqual(classify_attribute("winter running"), "use_case")


if __name__ == "__main__":
    unittest.main()
