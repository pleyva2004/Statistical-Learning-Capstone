import kagglehub
import os

# Specify your desired download directory
download_directory = "./data"

# Create the directory if it doesn't exist
os.makedirs(download_directory, exist_ok=True)

# Download latest version to specific directory
path = kagglehub.dataset_download("nikhil7280/student-performance-multiple-linear-regression", path=download_directory)

print("Path to dataset files:", path)