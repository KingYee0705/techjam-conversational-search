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
    def test_extended_catalog_vocabulary_classifies_common_attributes(self) -> None:
        self.assertEqual(classify_attribute("cashmere blend"), "material")
        self.assertEqual(classify_attribute("navy"), "color")
        self.assertEqual(classify_attribute("price around 75 dollars"), "budget")
        self.assertNotEqual(classify_attribute("necklace"), "material")

    def setUp(self) -> None:
        self.manager = DialogStateManager()
        self.manager.reset("session", PROFILE)

    @staticmethod
    def _manager_waiting_for(attribute: str) -> tuple[DialogStateManager, int]:
        manager = DialogStateManager(broad_question_limit=0)
        manager.reset("paraphrase", PROFILE)
        turn = 1
        decision = manager.process_turn(
            "paraphrase", "I'm looking for jackets, but I'm still exploring.", turn
        )
        while decision["ask_attribute"] != attribute:
            asked = decision["ask_attribute"]
            turn += 1
            decision = manager.process_turn(
                "paraphrase",
                f"I don't have an additional preference for {asked}.",
                turn,
            )
        return manager, turn + 1

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
        self.assertEqual(
            decision["constraint_priorities"],
            {"material": {"genuine leather": "hard"}},
        )
        self.assertIn("genuine leather", decision["search_query"])
        self.assertFalse(decision["is_vague"])

    def test_confirmed_constraint_priority_is_soft(self) -> None:
        self.manager.process_turn(
            "session",
            "I'm looking for winter boots. A key requirement is: genuine leather.",
            1,
        )
        decision = self.manager.process_turn(
            "session", "For that, what matters is: waterproof.", 2
        )

        self.assertEqual(
            decision["constraint_priorities"],
            {
                "material": {"genuine leather": "hard"},
                "feature": {"waterproof": "soft"},
            },
        )

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

    def test_multiple_same_attribute_details_trigger_one_bounded_follow_up(self) -> None:
        manager = DialogStateManager(broad_question_limit=0)
        manager.reset("specific", PROFILE)
        first = manager.process_turn(
            "specific", "I'm looking for jackets, but I'm still exploring.", 1
        )
        second = manager.process_turn(
            "specific",
            "For that, what matters is: waterproof; zippered pockets.",
            2,
        )
        third = manager.process_turn(
            "specific",
            "For that, what matters is: detachable hood; reflective trim.",
            3,
        )

        self.assertEqual(first["ask_attribute"], "feature")
        self.assertEqual(second["ask_attribute"], "feature")
        self.assertNotEqual(third["ask_attribute"], "feature")
        self.assertEqual(
            manager.get_state("specific")["asked_attributes"].count("feature"),
            2,
        )

    def test_mixed_attribute_details_do_not_repeat_the_question(self) -> None:
        manager = DialogStateManager(broad_question_limit=0)
        manager.reset("mixed", PROFILE)
        manager.process_turn(
            "mixed", "I'm looking for jackets, but I'm still exploring.", 1
        )
        decision = manager.process_turn(
            "mixed",
            "For that, what matters is: cotton; waterproof.",
            2,
        )

        self.assertNotEqual(decision["ask_attribute"], "feature")

    def test_declined_attribute_does_not_repeat_the_question(self) -> None:
        manager = DialogStateManager(broad_question_limit=0)
        manager.reset("declined", PROFILE)
        manager.process_turn(
            "declined", "I'm looking for jackets, but I'm still exploring.", 1
        )
        decision = manager.process_turn(
            "declined", "I don't have an additional preference for feature.", 2
        )

        self.assertNotEqual(decision["ask_attribute"], "feature")

    def test_no_preference_paraphrases_do_not_pollute_constraints(self) -> None:
        examples = (
            ("I don't mind the color.", "color"),
            ("I don\u2019t mind the color.", "color"),
            ("Any color is fine.", "color"),
            ("Any color works for me.", "color"),
            ("I'm open to any colour.", "color"),
            ("The color doesn't matter to me.", "color"),
            ("I'm flexible about color.", "color"),
            ("I'm not picky about the fabric.", "material"),
            ("The price is up to you.", "budget"),
            ("I have no strong preference regarding style.", "style"),
            ("Whatever material.", "material"),
            ("No particular preference.", "feature"),
        )
        for message, attribute in examples:
            with self.subTest(message=message):
                manager, turn = self._manager_waiting_for(attribute)
                decision = manager.process_turn("paraphrase", message, turn)
                state = manager.get_state("paraphrase")

                self.assertEqual(
                    decision["active_constraints"], {"category": ["jackets"]}
                )
                self.assertEqual(decision["search_query"], "jackets")
                self.assertIn(attribute, state["declined_attributes"])

    def test_additional_preference_paraphrase_preserves_existing_value(self) -> None:
        self.manager.process_turn(
            "session", "I'm looking for shirts. A key requirement is: cotton.", 1
        )
        decision = self.manager.process_turn(
            "session", "I have no additional preference about material.", 2
        )

        self.assertEqual(decision["active_constraints"]["material"], ["cotton"])
        self.assertEqual(
            decision["constraint_priorities"]["material"], {"cotton": "hard"}
        )
        self.assertIn("material", self.manager.get_state("session")["declined_attributes"])

    def test_general_no_preference_clears_existing_value(self) -> None:
        self.manager.process_turn(
            "session", "I'm looking for shirts. A key requirement is: cotton.", 1
        )
        decision = self.manager.process_turn(
            "session", "I have no particular preference about material.", 2
        )

        self.assertEqual(decision["active_constraints"], {"category": ["shirts"]})
        self.assertEqual(decision["constraint_priorities"], {})
        self.assertIn("material", self.manager.get_state("session")["declined_attributes"])

    def test_dont_mind_concrete_budget_is_not_a_decline(self) -> None:
        manager, turn = self._manager_waiting_for("budget")
        decision = manager.process_turn(
            "paraphrase", "I don't mind paying $50.", turn
        )
        state = manager.get_state("paraphrase")

        self.assertNotIn("budget", state["declined_attributes"])
        self.assertEqual(
            decision["active_constraints"]["budget"], ["I don't mind paying $50"]
        )

    def test_same_attribute_follow_up_never_overrides_turn_ten_boundary(self) -> None:
        manager = DialogStateManager(broad_question_limit=0)
        manager.reset("final", PROFILE)
        manager.process_turn(
            "final", "I'm looking for jackets, but I'm still exploring.", 1
        )
        decision = manager.process_turn(
            "final",
            "For that, what matters is: waterproof; zippered pockets.",
            10,
        )

        self.assertIsNone(decision["ask_attribute"])

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

    def test_retargeted_question_updates_the_pending_answer_field(self) -> None:
        manager = DialogStateManager(broad_question_limit=0)
        manager.reset("retarget", PROFILE)
        first = manager.process_turn(
            "retarget", "I'm looking for jackets, but I'm still exploring.", 1
        )
        self.assertEqual(first["ask_attribute"], "feature")

        message = manager.retarget_question("retarget", "color")
        state = manager.get_state("retarget")
        self.assertIn("color", message.lower())
        self.assertEqual(state["pending_attribute"], "color")
        self.assertEqual(state["asked_attributes"], ["color"])

        decision = manager.process_turn("retarget", "Blue.", 2)
        self.assertEqual(decision["active_constraints"]["color"], ["Blue"])

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
        self.assertEqual(decision["negative_constraints"], {})
        self.assertNotIn("red", decision["search_query"].lower())

    def test_direct_override_paraphrases_replace_stale_values(self) -> None:
        examples = (
            "Can you switch from red to blue?",
            "I am switching from red to blue.",
            "Please replace red with blue.",
            "I no longer want red; blue please.",
            "Now I want blue.",
            "Blue instead.",
            "I want blue instead.",
            "Instead, I want blue.",
        )
        for message in examples:
            with self.subTest(message=message):
                manager = DialogStateManager()
                manager.reset("override-paraphrase", PROFILE)
                manager.process_turn(
                    "override-paraphrase",
                    "I'm looking for walking shoes. color: red.",
                    1,
                )
                decision = manager.process_turn(
                    "override-paraphrase", message, 2
                )

                self.assertTrue(decision["is_override"])
                self.assertEqual(
                    [
                        value.lower()
                        for value in decision["active_constraints"]["color"]
                    ],
                    ["blue"],
                )
                self.assertNotIn("red", decision["search_query"].lower())

    def test_no_longer_without_replacement_removes_the_old_value(self) -> None:
        self.manager.process_turn(
            "session", "I'm looking for walking shoes. color: red.", 1
        )
        decision = self.manager.process_turn(
            "session", "I no longer want red.", 2
        )

        self.assertTrue(decision["is_override"])
        self.assertNotIn("color", decision["active_constraints"])
        self.assertNotIn("red", decision["search_query"].lower())

    def test_make_it_inside_an_explanation_is_not_an_override(self) -> None:
        examples = (
            "Reinforced stitching will make it durable.",
            "I want a zipper to make it easier.",
        )
        for message in examples:
            with self.subTest(message=message):
                manager = DialogStateManager()
                manager.reset("make-it", PROFILE)
                manager.process_turn("make-it", "I'm looking for jackets.", 1)
                decision = manager.process_turn("make-it", message, 2)

                self.assertFalse(decision["is_override"])
                self.assertIn("feature", decision["active_constraints"])

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
        self.assertEqual(decision["negative_constraints"]["material"], ["leather"])
        self.assertNotIn("leather", decision["search_query"].lower())

    def test_negative_preference_paraphrases_are_kept_out_of_search(self) -> None:
        examples = (
            ("anything but red", "color", "red"),
            ("except polyester", "material", "polyester"),
            ("steer clear of suede", "material", "suede"),
            ("I don't want nylon", "material", "nylon"),
            ("no leather please", "material", "leather"),
        )
        for phrase, attribute, value in examples:
            with self.subTest(phrase=phrase):
                manager = DialogStateManager()
                manager.reset("negative", PROFILE)
                decision = manager.process_turn(
                    "negative", f"I'm looking for jackets. {phrase}.", 1
                )

                self.assertNotIn(attribute, decision["active_constraints"])
                self.assertEqual(
                    decision["negative_constraints"][attribute],
                    [value],
                )
                self.assertNotIn(value, decision["search_query"].lower())

    def test_curly_apostrophe_negative_is_parsed_like_ascii(self) -> None:
        decision = self.manager.process_turn(
            "session", "I'm looking for jackets. I don’t want leather.", 1
        )

        self.assertNotIn("material", decision["active_constraints"])
        self.assertEqual(decision["negative_constraints"]["material"], ["leather"])

    def test_compound_negative_alternatives_are_stored_individually(self) -> None:
        decision = self.manager.process_turn(
            "session",
            "I'm looking for jackets. I don't want leather or wool.",
            1,
        )

        self.assertNotIn("material", decision["active_constraints"])
        self.assertEqual(
            decision["negative_constraints"]["material"],
            ["leather", "wool"],
        )
        self.assertNotIn("leather", decision["search_query"].lower())
        self.assertNotIn("wool", decision["search_query"].lower())

    def test_coordinated_and_filler_negatives_are_cleanly_excluded(self) -> None:
        examples = (
            ("Avoid leather and wool.", ["leather", "wool"]),
            ("Without any leather.", ["leather"]),
            ("Not made of leather.", ["leather"]),
        )
        for phrase, expected in examples:
            with self.subTest(phrase=phrase):
                manager = DialogStateManager()
                manager.reset("negative-fillers", PROFILE)
                decision = manager.process_turn(
                    "negative-fillers", f"I'm looking for jackets. {phrase}", 1
                )

                self.assertEqual(
                    decision["negative_constraints"]["material"], expected
                )
                self.assertNotIn("material", decision["active_constraints"])

    def test_decline_with_benign_but_clause_does_not_pollute_state(self) -> None:
        examples = (
            ("color", "Any color is fine, but show me some options."),
            ("material", "I don't mind, but thanks."),
        )
        for attribute, message in examples:
            with self.subTest(message=message):
                manager, turn = self._manager_waiting_for(attribute)
                decision = manager.process_turn("paraphrase", message, turn)
                state = manager.get_state("paraphrase")

                self.assertNotIn(attribute, decision["active_constraints"])
                self.assertIn(attribute, state["declined_attributes"])

    def test_any_attribute_but_value_keeps_only_the_exclusion(self) -> None:
        manager, turn = self._manager_waiting_for("color")
        decision = manager.process_turn(
            "paraphrase", "Any color but red.", turn
        )

        self.assertNotIn("color", decision["active_constraints"])
        self.assertEqual(decision["negative_constraints"]["color"], ["red"])

    def test_concrete_information_is_not_swallowed_as_a_decline(self) -> None:
        manager, turn = self._manager_waiting_for("color")
        decision = manager.process_turn(
            "paraphrase", "I don't mind the color being blue.", turn
        )
        self.assertEqual(decision["active_constraints"]["color"], ["blue"])
        self.assertNotIn("color", manager.get_state("paraphrase")["declined_attributes"])

        material_manager, material_turn = self._manager_waiting_for("material")
        material_decision = material_manager.process_turn(
            "paraphrase", "Whatever material keeps me warm.", material_turn
        )
        self.assertIn("material", material_decision["active_constraints"])
        self.assertNotIn(
            "material",
            material_manager.get_state("paraphrase")["declined_attributes"],
        )

    def test_negative_and_positive_compound_clauses_are_separated(self) -> None:
        for separator in (", but ", " but ", "; "):
            with self.subTest(separator=separator):
                manager = DialogStateManager()
                manager.reset("mixed-clauses", PROFILE)
                decision = manager.process_turn(
                    "mixed-clauses",
                    f"I'm looking for jackets. No leather{separator}waterproof.",
                    1,
                )

                self.assertEqual(decision["category"], "jackets")
                self.assertEqual(
                    decision["negative_constraints"]["material"], ["leather"]
                )
                self.assertEqual(
                    decision["active_constraints"]["feature"], ["waterproof"]
                )
                self.assertNotIn("leather", decision["search_query"].lower())

    def test_category_clause_stops_before_negative_exception(self) -> None:
        decision = self.manager.process_turn(
            "session", "I'm looking for shoes but not boots.", 1
        )

        self.assertEqual(decision["category"], "shoes")
        self.assertEqual(decision["negative_constraints"]["category"], ["boots"])
        self.assertEqual(decision["search_query"], "shoes")

    def test_standalone_no_additional_preference_is_a_decline(self) -> None:
        manager, turn = self._manager_waiting_for("feature")
        decision = manager.process_turn(
            "paraphrase", "No additional preference.", turn
        )

        self.assertNotIn("feature", decision["active_constraints"])
        self.assertEqual(decision["negative_constraints"], {})
        self.assertIn(
            "feature", manager.get_state("paraphrase")["declined_attributes"]
        )

    def test_standalone_no_additional_preference_can_name_an_attribute(self) -> None:
        manager = DialogStateManager()
        manager.reset("additional-target", PROFILE)
        manager.process_turn(
            "additional-target",
            "I'm looking for shirts. A key requirement is: cotton.",
            1,
        )
        decision = manager.process_turn(
            "additional-target", "No additional preference for material.", 2
        )

        self.assertEqual(decision["active_constraints"]["material"], ["cotton"])
        self.assertIn(
            "material",
            manager.get_state("additional-target")["declined_attributes"],
        )

    def test_pending_answers_preserve_qualifiers_and_mixed_constraints(self) -> None:
        examples = (
            ("other", "blue and waterproof", {"color": ["blue"], "feature": ["waterproof"]}),
            ("color", "dark blue", {"color": ["dark blue"]}),
            ("material", "lightweight cotton", {"material": ["lightweight cotton"]}),
        )
        for attribute, message, expected in examples:
            with self.subTest(message=message):
                manager, turn = self._manager_waiting_for(attribute)
                decision = manager.process_turn("paraphrase", message, turn)
                for expected_attribute, values in expected.items():
                    self.assertEqual(
                        decision["active_constraints"][expected_attribute], values
                    )

    def test_pending_category_preserves_positive_part_and_exclusion(self) -> None:
        for message in ("shoes, not boots", "shoes without boots"):
            with self.subTest(message=message):
                manager = DialogStateManager()
                manager.reset("pending-category", PROFILE)
                first = manager.process_turn("pending-category", "", 1)
                self.assertEqual(first["ask_attribute"], "category")
                decision = manager.process_turn("pending-category", message, 2)

                self.assertEqual(decision["category"], "shoes")
                self.assertEqual(
                    decision["negative_constraints"]["category"], ["boots"]
                )

    def test_decline_accepts_common_attribute_noun_phrases(self) -> None:
        examples = (
            ("size", "No preference for shoe size."),
            ("color", "No preference for the color of the item."),
            ("brand", "No preference regarding brand name."),
        )
        for attribute, message in examples:
            with self.subTest(message=message):
                manager, turn = self._manager_waiting_for(attribute)
                decision = manager.process_turn("paraphrase", message, turn)

                self.assertNotIn(attribute, decision["active_constraints"])
                self.assertIn(
                    attribute,
                    manager.get_state("paraphrase")["declined_attributes"],
                )

    def test_benign_but_clause_must_not_hide_a_concrete_value(self) -> None:
        manager, turn = self._manager_waiting_for("color")
        decision = manager.process_turn(
            "paraphrase", "Any color is fine, but show me blue options.", turn
        )

        self.assertEqual(decision["active_constraints"]["color"], ["blue"])
        self.assertNotIn(
            "color", manager.get_state("paraphrase")["declined_attributes"]
        )

    def test_no_more_than_budget_is_not_misread_as_a_negative_value(self) -> None:
        decision = self.manager.process_turn(
            "session",
            "I'm looking for jackets. A key requirement is: budget no more than $50.",
            1,
        )

        self.assertEqual(
            decision["active_constraints"]["budget"],
            ["budget no more than $50"],
        )
        self.assertEqual(decision["negative_constraints"], {})

    def test_structured_negative_requirement_is_not_stored_as_positive(self) -> None:
        decision = self.manager.process_turn(
            "session",
            "I'm looking for jackets. A key requirement is: not leather.",
            1,
        )

        self.assertNotIn("material", decision["active_constraints"])
        self.assertEqual(decision["excluded_constraints"]["material"], ["leather"])
        self.assertEqual(decision["negative_constraints"]["material"], ["leather"])
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

    def test_obvious_budget_formats_support_commas_ranges_and_minimums(self) -> None:
        examples = (
            ("Under $1,000.", "under $1,000"),
            ("Budget between 500 and 1,000.", "budget between 500 and 1,000"),
            ("From $500 to $1,000.", "from $500 to $1,000"),
            ("At least $500.", "at least $500"),
            ("Budget minimum of 500.", "budget minimum of 500"),
        )
        for message, expected in examples:
            with self.subTest(message=message):
                manager = DialogStateManager()
                manager.reset("budget-format", PROFILE)
                decision = manager.process_turn("budget-format", message, 1)

                self.assertEqual(
                    [value.lower() for value in decision["active_constraints"]["budget"]],
                    [expected],
                )

    def test_size_under_ten_is_not_classified_as_a_budget(self) -> None:
        self.assertEqual(classify_attribute("shoe size under 10"), "size")
        self.assertEqual(classify_attribute("width under 10"), "size")
        self.assertEqual(classify_attribute("under $10"), "budget")

        manager = DialogStateManager()
        manager.reset("size-limit", PROFILE)
        decision = manager.process_turn(
            "size-limit", "I need a walking shoe size under 10.", 1
        )
        self.assertNotIn("budget", decision["active_constraints"])
        self.assertEqual(
            decision["active_constraints"]["size"], ["shoe size under 10"]
        )

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
