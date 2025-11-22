import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join("src", "components", "data", "raw.csv")
        self.train_data_path = os.path.join("src", "components", "data", "train.csv")
        self.test_data_path = os.path.join("src", "components", "data", "test.csv")

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")

        try:
            # Read your dataset here
            df = pd.read_csv(self.raw_data_path)
            logging.info("Read the raw dataset")

            # Train test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train-test split completed")

            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)

            train_set.to_csv(self.train_data_path, index=False, header=True)
            test_set.to_csv(self.test_data_path, index=False, header=True)

            logging.info("Ingestion completed successfully")

            return (
                self.train_data_path,
                self.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
