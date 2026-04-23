import unittest

from src.nlp_bilstm import clean_text, weak_issue_label, weak_sentiment_label


class TestNlpBiLstmHelpers(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(clean_text("Billing issue!!! Slow network."), "billing issue slow network")

    def test_weak_issue_label(self):
        row = {"PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "support_calls": 1}
        self.assertEqual(weak_issue_label(row), "billing")

    def test_weak_sentiment_label(self):
        row = {"support_calls": 4, "Churn": "Yes"}
        self.assertEqual(weak_sentiment_label(row), "negative")


if __name__ == "__main__":
    unittest.main()
