import unittest
import joblib
from sklearn.ensemble import RandomForestClassifier

class TestChurnModel(unittest.TestCase):
    def test_model_type_and_features(self):
        # Load the saved model
        model = joblib.load('models/churn_model.pkl')

        # Check model type
        self.assertIsInstance(model, RandomForestClassifier)

        # Check feature importances length (should match number of input features)
        self.assertGreaterEqual(len(model.feature_importances_), 10)  # Adjust based on actual feature count

if __name__ == '__main__':
    unittest.main()
