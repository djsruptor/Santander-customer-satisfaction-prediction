import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_ID = 46859
RAW_FILE = os.path.join(PROJECT_ROOT, 'data', 'santander_raw.parquet')
REDUCED_FILE = os.path.join(PROJECT_ROOT, 'data', 'santander_reduced.parquet')

MODEL_FILE = os.path.join(PROJECT_ROOT, 'models', 'best_model.json')