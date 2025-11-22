
        
        
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str = "artifacts/data_ingestion/raw.csv"
    processed_data_path: str = "artifacts/data_ingestion/processed.csv"
    train_data_path: str = "artifacts/data_ingestion/train.csv"
    test_data_path: str = "artifacts/data_ingestion/test.csv"


class DataIngestion:

    def __init__(self, input_path):
        self.input_path = input_path
        self.config = DataIngestionConfig()

        # Create artifact folder
        os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

    def initiate_data_ingestion(self):
        print("📥 Reading input CSV...")
        df = pd.read_csv(self.input_path)

        # Save raw file
        df.to_csv(self.config.raw_data_path, index=False)
        print(f"✅ Raw file saved at: {self.config.raw_data_path}")

        # ----------- Processing (optional) -----------
        df_processed = df.drop_duplicates()
        df_processed.to_csv(self.config.processed_data_path, index=False)
        print(f"✅ Processed file saved at: {self.config.processed_data_path}")

        # ----------- Train-Test Split --------------
        train, test = train_test_split(df_processed, test_size=0.2, random_state=42)

        train.to_csv(self.config.train_data_path, index=False)
        test.to_csv(self.config.test_data_path, index=False)

        print(f"📊 Train file saved at: {self.config.train_data_path}")
        print(f"📊 Test file saved at: {self.config.test_data_path}")

        return (
            self.config.raw_data_path,
            self.config.processed_data_path,
            self.config.train_data_path,
            self.config.test_data_path
        )

