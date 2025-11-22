import os
import shutil
from components.data.data_ingestion import DataIngestion

# Path to your current dataset
current_dataset = r"C:\Users\Vaishnavi Jaiswal\OneDrive\New folder\New folder\ML Project\src\components\data\stud.csv"

# Destination path required by DataIngestion
data_folder = r"C:\Users\Vaishnavi Jaiswal\OneDrive\New folder\New folder\ML Project\src\components\data"
raw_file = os.path.join(data_folder, "raw.csv")

os.makedirs(data_folder, exist_ok=True)
shutil.copy(current_dataset, raw_file)
print(f"Dataset copied/renamed to: {raw_file}")

# Run data ingestion
obj = DataIngestion()
train_path, test_path = obj.initiate_data_ingestion()

print(f"✅ Training data saved at: {train_path}")
print(f"✅ Testing data saved at: {test_path}")
