import pandas as pd
import numpy as np
import unittest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import analysis


# Refactared by deleting clean_data and analyze_data
def load_path(file_path):
    """Loads data from a given text file"""
    with open(file_path, "r") as file:
        path = file.read().strip().split("\n")
    return path


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "Age": [25, 30, 35, None, 150],
                "BMI": [22.5, 27.8, None, 30.0, 45.0],
                "Sleep_Hours": [7, 8, 6, -1, 25],
                "Heart_Rate": [70, 75, 80, 200, -10],
            }
        )

    # refactored by using extract method
    def get_clean_data(self):
        return analysis.clean_data(self.df)

    def test_clean_data(self):
        cleaned_df = self.get_clean_data()
        self.assertEqual(len(cleaned_df), 3)
        self.assertTrue((cleaned_df["Age"].between(10, 100)).all())
        self.assertTrue((cleaned_df["Sleep_Hours"].between(0, 24)).all())
        self.assertTrue((cleaned_df["BMI"].between(0, 40)).all())
        self.assertTrue((cleaned_df["Heart_Rate"].between(0, 120)).all())

    def test_analyze_data(self):
        cleaned_df = self.get_clean_data()
        analyzis = analysis.analyze_data(cleaned_df)
        self.assertAlmostEqual(analyzis["average_age"], (25 + 30 + 35) / 3)
        expected_bmi = (22.5 + 27.8 + 28.9) / 3
        self.assertAlmostEqual(analyzis["average_bmi"], expected_bmi)
        exp_slp_hr = (7 + 8 + 6) / 3
        self.assertAlmostEqual(analyzis["average_sleep_hours"], exp_slp_hr)
        exp_hr = (70 + 75 + 80) / 3
        self.assertAlmostEqual(analyzis["average_heart_rate"], exp_hr)

    def test_model_training(self):
        cleaned_df = self.get_clean_data()
        X = cleaned_df[["Age", "BMI", "Sleep_Hours"]]
        y = cleaned_df["Heart_Rate"]
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        self.assertEqual(len(predictions), len(y))
        for pred in predictions:
            self.assertTrue(isinstance(pred, float).all())

    def test_end_to_end_flow(self):
        df = pd.DataFrame(
            {
                "Age": np.random.randint(20, 60, 100),
                "Caffeine_mg": np.random.randint(50, 300, 100),
                "Coffee_Intake": np.random.randint(0, 5, 100),
                "BMI": np.random.uniform(18, 35, 100),
                "Sleep_Hours": np.random.uniform(4, 9, 100),
                "Heart_Rate": np.random.randint(50, 100, 100),
                "Physical_Activity_Hours": np.random.uniform(0, 5, 100),
            }
        )
        # refactored by changing df_cleaned variable name to df_clean
        df_clean = analysis.clean_data(df)

        self.assertFalse(df_clean.isnull().values.any())
        self.assertTrue((df_clean["BMI"] <= 40).all())

        cols = ["Age", "Caffeine_mg", "Coffee_Intake", "BMI", "Sleep_Hours"]
        X = df_clean[cols]
        y = df_clean[["Heart_Rate"]]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        self.assertIsInstance(model, LinearRegression)
        self.assertGreaterEqual(mse, 0)
        self.assertGreaterEqual(r2, -1)
        self.assertLessEqual(r2, 1)


if __name__ == "__main__":
    unittest.main()
