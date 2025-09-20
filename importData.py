# Install dependencies as needed:
# pip install kagglehub pandas
import kagglehub
import pandas as pd
import os

# Download the dataset
path = kagglehub.dataset_download("nikhil7280/student-performance-multiple-linear-regression")

# Write the path to a .env file
with open('.env', 'w') as f:
    f.write(f"DATA_PATH={path}\n")


print("Dataset downloaded to:", path)

# List files in the downloaded directory
files = os.listdir(path)
print("Files in dataset:", files)

# Load the CSV file (assuming there's a CSV file in the dataset)
# We'll find the CSV file automatically
csv_files = [f for f in files if f.endswith('.csv')]

if csv_files:
    # Load the first CSV file found
    csv_file = csv_files[0]
    df = pd.read_csv(os.path.join(path, csv_file))
    print(f"\nLoaded {csv_file}")
    print("Dataset shape:", df.shape)
    print("\nFirst 5 records:")
    print(df.head())
    print("\nColumn names:")
    print(df.columns.tolist())
else:
    print("No CSV files found in the dataset")
    print("Available files:", files)