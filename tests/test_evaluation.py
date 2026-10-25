import unittest

from app.evaluation.dataset import validate_dataset
from app.evaluation.metrics import (
    aggregate_abstention, answer_fact_score, citation_metrics, retrieval_metrics,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_recall_and_mrr_respect_rank_and_k(self):
        expected = [{"source": "a.pdf", "page": 2}, {"source": "b.md", "page": None}]
        ranked = [
            {"source": "noise.md", "page": None},
            {"source": "b.md", "page": None},
        ]
        self.assertEqual(retrieval_metrics(expected, ranked, k=2), {"recall_at_2": 0.5, "mrr": 0.5})

    def test_citation_precision_and_coverage(self):
        expected = [{"source": "guide.pdf", "page": 3}, {"source": "policy.md", "page": None}]
        cited = [{"title": "guide.pdf — page 3"}, {"title": "unrelated.md"}]
        self.assertEqual(citation_metrics(expected, cited), {"citation_precision": 0.5, "citation_coverage": 0.5})

    def test_answer_facts_and_abstention(self):
        self.assertEqual(answer_fact_score(["MFA", "restart client"], "Enable MFA and restart client"), 1.0)
        self.assertEqual(aggregate_abstention([
            {"should_abstain": True, "abstained": True},
            {"should_abstain": False, "abstained": True},
        ]), {"abstention_precision": 0.5, "abstention_recall": 1.0, "abstention_accuracy": 0.5})

    def test_empty_denominators_are_unmeasured(self):
        self.assertEqual(retrieval_metrics([], [], 5), {"recall_at_5": None, "mrr": None})
        self.assertIsNone(answer_fact_score([], "anything"))


class EvaluationDatasetTests(unittest.TestCase):
    def valid_dataset(self):
        return {"dataset_version": "test", "cases": [{
            "id": "case-1", "question": "question", "expected_facts": [],
            "expected_evidence": [], "should_abstain": True,
            "expected_source_used": "insufficient_evidence", "offline": {"retrieved_chunks": []},
        }]}

    def test_valid_dataset(self):
        validate_dataset(self.valid_dataset())

    def test_duplicate_case_ids_are_rejected(self):
        data = self.valid_dataset()
        data["cases"].append(dict(data["cases"][0]))
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_dataset(data)

    def test_invalid_expected_evidence_page_is_rejected(self):
        data = self.valid_dataset()
        data["cases"][0]["expected_evidence"] = [{"source": "doc", "page": "second"}]
        with self.assertRaisesRegex(ValueError, "numeric"):
            validate_dataset(data)


if __name__ == "__main__":
    unittest.main()
