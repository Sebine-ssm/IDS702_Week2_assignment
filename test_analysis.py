import pandas as pd
import numpy as np
import unittest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  # For regression metrics
from analysis import clean_data, analyze_data

def load_path(file_path):
    with open(file_path, "r") as file:
        path = file.read().strip().split("\n")
    return path


def clean_data(df):
    df["BMI"] = df["BMI"].fillna(df["BMI"].median())
    df = df.dropna()
    df = df.drop_duplicates(keep="first")
    df = df[df["Age"].between(10, 100)]
    df = df[df["Sleep_Hours"].between(0, 24)]
    df = df[df["BMI"].between(0, 40)]
    df = df[df["Heart_Rate"].between(0, 120)]
    return df


def analyze_data(df):
    analysis = {}
    analysis["average_age"] = df["Age"].mean()
    analysis["average_bmi"] = df["BMI"].mean()
    analysis["average_sleep_hours"] = df["Sleep_Hours"].mean()
    analysis["average_heart_rate"] = df["Heart_Rate"].mean()
    return analysis


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

    def test_clean_data(self):
        cleaned_df = clean_data(self.df)
        self.assertEqual(len(cleaned_df), 3)
        self.assertTrue(all(cleaned_df["Age"].between(10, 100)))
        self.assertTrue(all(cleaned_df["Sleep_Hours"].between(0, 24)))
        self.assertTrue(all(cleaned_df["BMI"].between(0, 40)))
        self.assertTrue(all(cleaned_df["Heart_Rate"].between(0, 120)))

    def test_analyze_data(self):
        cleaned_df = clean_data(self.df)
        analysis = analyze_data(cleaned_df)
        self.assertAlmostEqual(analysis["average_age"], (25 + 30 + 35) / 3)
        expected_bmi = (22.5 + 27.8 + 28.9) / 3  
        self.assertAlmostEqual(
            analysis["average_bmi"], expected_bmi
        )  
        self.assertAlmostEqual(analysis["average_sleep_hours"], (7 + 8 + 6) / 3)
        self.assertAlmostEqual(analysis["average_heart_rate"], (70 + 75 + 80) / 3)

    def test_model_training(self):
        cleaned_df = clean_data(self.df)
        X = cleaned_df[["Age", "BMI", "Sleep_Hours"]]
        y = cleaned_df["Heart_Rate"]
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        self.assertEqual(len(predictions), len(y))
        self.assertTrue(all(isinstance(pred, float) for pred in predictions))

    def test_end_to_end_flow(self):
        # Create synthetic data inline
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

        df_cleaned = clean_data(df)

        self.assertFalse(df_cleaned.isnull().values.any())
        self.assertTrue((df_cleaned["BMI"] <= 40).all())

        X = df_cleaned[["Age", "Caffeine_mg", "Coffee_Intake", "BMI", "Sleep_Hours"]]
        y = df_cleaned[["Heart_Rate"]]
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