import os
import joblib
import pandas as pd

class PredictPipeline:
    def __init__(self):
        # Correct path to artifacts folder in project root
        base_path = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
        self.model_path = os.path.abspath(os.path.join(base_path, "model.pkl"))
        self.scaler_path = os.path.abspath(os.path.join(base_path, "scaler.pkl"))

        # Check if files exist
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {self.scaler_path}")

    def predict(self, features: pd.DataFrame):
        try:
            # Load model and scaler
            model = joblib.load(self.model_path)
            scaler = joblib.load(self.scaler_path)

            # Scale the input features
            scaled_features = scaler.transform(features)

            # Make predictions
            preds = model.predict(scaled_features)
            return preds

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")


class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            data_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to create DataFrame: {e}")


# ✅ Test block
if __name__ == "__main__":
    # Example input
    data = CustomData(
        gender="female",
        race_ethnicity="group B",
        parental_level_of_education="bachelor's degree",
        lunch="standard",
        test_preparation_course="completed",
        reading_score=72,
        writing_score=74
    )

    df = data.get_data_as_dataframe()
    print("Input DataFrame:\n", df)

    # Make prediction
    predict_pipeline = PredictPipeline()
    prediction = predict_pipeline.predict(df)
    print("Prediction:", prediction)
