import os
import shutil
import sys

# Add src folder to the Python path
sys.path.append(os.path.abspath("src"))

from pipeline.data_ingestion import DataIngestion

# File paths
input_csv = "src/components/data/stud.csv"
raw_csv = "src/components/data/raw.csv"

# Make sure data folder exists
os.makedirs("src/components/data", exist_ok=True)

# Copy stud.csv to raw.csv
shutil.copy(input_csv, raw_csv)
print(f"Dataset copied/renamed to: {raw_csv}")

# Create ingestion object (ONLY input_path)
ingestion = DataIngestion(input_path=raw_csv)

# Start ingestion
result = ingestion.initiate_data_ingestion()

print("✔ Data ingestion completed!")
print(f"Files saved at: {result}")
