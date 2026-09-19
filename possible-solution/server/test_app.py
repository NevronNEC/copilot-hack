import importlib
import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SERVER_DIR = Path(__file__).parent


class FakeModel:
    def __init__(self):
        self.inputs = None

    def predict_proba(self, inputs):
        self.inputs = inputs
        return [FakePrediction()]


class FakePrediction:
    def __str__(self):
        return "[0.25 0.75]"


class AppRoutesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_cwd = Path.cwd()
        os.chdir(SERVER_DIR)
        sys.path.insert(0, str(SERVER_DIR))
        cls.model = FakeModel()

        # Importing app loads the model at module scope. Keep this test
        # independent of the binary model and its optional dependencies.
        with patch("pickle.load", return_value=cls.model), patch(
            "builtins.open", return_value=io.BytesIO()
        ):
            cls.app = importlib.import_module("app")

        cls.client = cls.app.app.test_client()

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.original_cwd)
        sys.path.remove(str(SERVER_DIR))

    def test_airports_returns_sorted_airports_with_numeric_ids(self):
        response = self.client.get("/airports")

        self.assertEqual(response.status_code, 200)
        airports = response.get_json()
        self.assertIsInstance(airports, list)
        self.assertTrue(airports)
        self.assertEqual(
            airports,
            sorted(airports, key=lambda airport: airport["name"]),
        )
        self.assertTrue(all(isinstance(airport["id"], int) for airport in airports))
        self.assertTrue(all(set(airport) == {"id", "name"} for airport in airports))

    def test_predict_converts_query_values_and_returns_probability_fields(self):
        response = self.client.get("/predict?day_of_week=3&airport_id=14771")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.model.inputs, [[3, 14771]])
        self.assertEqual(response.get_json(), {"certainty": 0.25, "delay": 0.75})
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")

    def test_predict_rejects_missing_query_values(self):
        response = self.client.get("/predict?day_of_week=3")

        self.assertEqual(response.status_code, 500)


if __name__ == "__main__":
    unittest.main()
