import joblib
import os
import uvicorn
import pandas as pd
import json
from fastapi import FastAPI, HTTPException
from typing import Dict
from pydantic import BaseModel
import pyarrow.parquet as pq
from src.config import MODEL_FILE, REDUCED_FILE
from src.data import json_customer

os.makedirs("examples", exist_ok=True)

table = pq.read_table(REDUCED_FILE, use_threads=True)
df = table.slice(0, 1000).to_pandas()

with open(MODEL_FILE, 'rb') as f_in:
    metadata = joblib.load(f_in)

model = metadata['model']
feature_names = metadata['feature_names']

if 'TARGET' in df.columns:
    df = df.drop(columns='TARGET')

app = FastAPI(title='Customer dissatisfaction classification 2')

def predict_single(customer: Dict[str, float]) -> float:
    df = pd.DataFrame([customer], columns=feature_names)
    prob = model.predict_proba(df)[0, 1]
    return float(prob)

class PredictResponse(BaseModel):
    dissatisfaction_probability: float
    dissatisfied: bool

@app.get(
    '/files',
    summary='Generate JSON for indexed customer',
    description='Creates a customer JSON file you can later use for prediction. Top 1000 customers are available for testing.'
)
def generate_sample(index: int = 0):
    if index < 0 or index >= len(df):
        raise HTTPException(404, "Index out of range")
    data = json_customer(df, index)
    return {'message': f'sample {index} generated', 'file': data}

@app.post(
        '/predict',
        summary='Predict disatisfaction for customer by index number',
        description='Loads a JSON file from the examples folder based on the index number entered, if it does not exist, generate it using the GET first'
        )
def predict_from_file(number: int) -> PredictResponse:
    file_path = f'examples/{number}.json'

    if not os.path.exists(file_path):
        raise HTTPException(404, f"File {file_path} not found. Use GET /files first.")

    with open(file_path, 'r') as f:
        data = json.load(f)

    prob = predict_single(data)
    return PredictResponse(
        dissatisfaction_probability=prob,
        dissatisfied=bool(prob > 0.136)
    )

if __name__ == '__main__':
    port = int(os.getenv("PORT", 8000))
    uvicorn.run('predict:app', host='0.0.0.0', port=port, reload=True)