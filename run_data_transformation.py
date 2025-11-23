import os
import sys

# Add src folder to Python path
src_path = os.path.join(os.path.dirname(__file__), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from components.data.data_ingestion import DataIngestion
from components.data_transformation.data_transformation import DataTransformation

# File paths
train_csv = os.path.join("artifacts", "data_ingestion", "train.csv")
test_csv = os.path.join("artifacts", "data_ingestion", "test.csv")

# Ensure artifacts folder exists
os.makedirs("artifacts", exist_ok=True)

# Initialize Data Transformation
data_transformation = DataTransformation()

# Start transformation
train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
    train_path=train_csv,
    test_path=test_csv
)

print("✅ Data transformation completed!")
print("Transformed train and test arrays ready.")
print(f"Preprocessor saved at: {preprocessor_path}")
