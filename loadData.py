import pandas as pd
import os
from dotenv import load_dotenv


def load_data():
    load_dotenv()

    path = os.environ["DATA_PATH"]

    # Find CSV files in the directory
    files = os.listdir(path)
    csv_files = [f for f in files if f.endswith('.csv')]

    if csv_files:
        # Load the first CSV file found
        csv_file = csv_files[0]
        csv_path = os.path.join(path, csv_file)
        df = pd.read_csv(csv_path)
        print(f"Loaded {csv_file}")
        print(df.head())
    else:
        print("No CSV files found in the dataset")
        print("Available files:", files)
    
    return df