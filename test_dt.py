import os
import unittest

import joblib
import pandas as pd


BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "models")


class TestDecisionTreeArtifact(unittest.TestCase):
    def test_decision_tree_predict_proba_shape(self):
        dt_model = joblib.load(os.path.join(MODELS_DIR, "churn_dt_model.joblib"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
        feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))

        sample = pd.DataFrame([{col: 0 for col in feature_cols}])[feature_cols]
        scaled = scaler.transform(sample)
        proba = dt_model.predict_proba(scaled)

        self.assertEqual(proba.shape, (1, 2))
        self.assertGreaterEqual(proba[0][1], 0.0)
        self.assertLessEqual(proba[0][1], 1.0)


if __name__ == "__main__":
    unittest.main()
