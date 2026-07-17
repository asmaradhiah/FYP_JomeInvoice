import unittest

from evaluation_metrics import (
    compute_context_precision,
    compute_context_recall,
    compute_hit_at_1,
    compute_hit_at_3,
    compute_hit_at_5,
)


class EvaluationMetricsTests(unittest.TestCase):
    def test_hit_at_1_is_zero_when_the_first_item_is_not_relevant(self):
        items = [
            {"text": "unrelated passage"},
            {"text": "this mentions the invoice payment details"},
        ]
        self.assertEqual(compute_hit_at_1(items, "payment details", "Who pays?"), 0.0)

    def test_hit_at_3_is_one_when_any_top3_context_is_relevant(self):
        items = [
            {"text": "unrelated passage"},
            {"text": "this mentions the invoice payment details"},
        ]
        self.assertEqual(compute_hit_at_3(items, "payment details", "Who pays?"), 1.0)

    def test_hit_at_5_is_zero_when_no_top5_context_is_relevant(self):
        items = [
            {"text": "unrelated passage"},
            {"text": "another unrelated text"},
        ]
        self.assertEqual(compute_hit_at_5(items, "payment details", "Who pays?"), 0.0)

    def test_context_precision_counts_relevant_items_in_top5(self):
        items = [
            {"text": "payment details for invoice"},
            {"text": "unrelated passage"},
            {"text": "invoice payment process"},
        ]
        self.assertAlmostEqual(compute_context_precision(items, "payment details", "Who pays?", k=5), 2 / 3)

    def test_context_recall_counts_relevant_tokens_in_top5(self):
        items = [
            {"text": "payment details for invoice"},
            {"text": "unrelated passage"},
        ]
        self.assertAlmostEqual(compute_context_recall(items, "payment details", "Payment details", k=5), 1.0)


if __name__ == "__main__":
    unittest.main()
