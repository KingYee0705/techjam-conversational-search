from __future__ import annotations

import unittest

import numpy as np

from starter.embedding_retrieval import (
    EmbeddingRetriever,
    product_document,
    semantic_query,
)


class _FakeModel:
    def encode_query(self, text: str, **kwargs: object) -> np.ndarray:
        del text, kwargs
        return np.asarray([1.0, 0.0], dtype=np.float32)


class EmbeddingRetrievalTest(unittest.TestCase):
    def test_product_document_prioritizes_searchable_fields(self) -> None:
        document = product_document({
            "title": "Cushioned walking shoes",
            "categories": ["Shoes", "Walking"],
            "features": ["arch support"],
            "details": {"Material": "mesh"},
            "description": ["Comfort for long days"],
            "store": "Example",
        })

        self.assertTrue(document.startswith("title: Cushioned walking shoes"))
        self.assertIn("features: arch support", document)
        self.assertIn("Material: mesh", document)

    def test_semantic_query_combines_current_state_deterministically(self) -> None:
        query = semantic_query(
            "Shoes for standing all day",
            "Walking Shoes",
            {"material": ["mesh"], "feature": ["arch support"]},
        )

        self.assertEqual(
            query,
            "Shoes for standing all day. category: Walking Shoes. "
            "feature: arch support. material: mesh",
        )

    def test_retrieve_orders_cosine_scores_and_respects_top_k(self) -> None:
        retriever = EmbeddingRetriever.__new__(EmbeddingRetriever)
        retriever._np = np
        retriever.model = _FakeModel()
        retriever.parent_asins = ("B", "A", "C")
        retriever.embeddings = np.asarray(
            [[0.8, 0.2], [0.8, -0.2], [0.1, 0.9]], dtype=np.float32
        )

        results = retriever.retrieve("walking", top_k=2)

        self.assertEqual([item["parent_asin"] for item in results], ["A", "B"])
        self.assertTrue(all(item["route_hits"] == ["semantic"] for item in results))


if __name__ == "__main__":
    unittest.main()
