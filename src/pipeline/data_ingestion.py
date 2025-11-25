import os
import sys
import pandas as pd

class DataIngestion:
    def __init__(self, input_path="src/components/data/raw.csv"):
        self.input_path = input_path
        self.train_path = os.path.join("artifacts", "train.csv")
        self.test_path = os.path.join("artifacts", "test.csv")

    def initiate_data_ingestion(self):
        df = pd.read_csv(self.input_path)

        # Create artifacts folder if not exists
        os.makedirs("artifacts", exist_ok=True)

        # Train-test split
        from sklearn.model_selection import train_test_split
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

        # Save files
        train_set.to_csv(self.train_path, index=False, header=True)
        test_set.to_csv(self.test_path, index=False, header=True)

        return {
            "train_path": self.train_path,
            "test_path": self.test_path,
        }
