import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import openml
import pandas as pd
from src.config import DATASET_ID, RAW_FILE

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

openml.config.cache_directory = 'data/cache'

def download_data(dataset_id=46859):
    
    if os.path.exists(RAW_FILE):
        print('File already exists')
        return

    print('Downloading file from OpenML...')
    dataset = openml.datasets.get_dataset(DATASET_ID)
    df, *_ = dataset.get_data()
    
    df.to_parquet(RAW_FILE, index=False)
    print('File saved!')

if __name__ == "__main__":
    download_data()