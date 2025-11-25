import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

class DataTransformation:
    def initiate_data_transformation(self, train_path, test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train = train_df.drop("average", axis=1)
        y_train = train_df["average"]

        X_test = test_df.drop("average", axis=1)
        y_test = test_df["average"]

        num_cols = X_train.columns.tolist()

        pipeline = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols)
            ]
        )

        X_train = pipeline.fit_transform(X_train)
        X_test = pipeline.transform(X_test)

        return X_train, X_test, y_train, y_test
