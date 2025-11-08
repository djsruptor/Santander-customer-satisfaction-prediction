import pickle
import os
import uvicorn
import pandas as pd
import json
from fastapi import FastAPI
from typing import Dict
from pydantic import BaseModel
from src.data import get_data, remove_correlated_features, json_customer

model_path = 'models/best_model.pkl'
with open(model_path, 'rb') as f_in:
    metadata = pickle.load(f_in)

model = metadata['model']
feature_names = metadata['feature_names']

df = get_data()
df_reduced, dropped = remove_correlated_features(df)
if 'TARGET' in df_reduced.columns:
    df_reduced = df_reduced.drop(columns='TARGET')

app = FastAPI(title='Customer dissatisfaction classification 2')

def predict_single(customer: Dict[str, float]) -> float:
    df = pd.DataFrame([customer], columns=feature_names)
    result = model.predict_proba(df)[0, 1]
    return float(result)

class PredictResponse(BaseModel):
    dissatisfaction_probability: float
    dissatisfied: bool

@app.get(
    '/files',
    summary='Generate JSON for indexed customer',
    description='Creates a customer JSON file you can later use for prediction.'
)
def generate_sample(index: int = 0):
    data = json_customer(df_reduced, index)
    return {'message': f'sample {index} generated', 'sample': data}

@app.post(
        '/predict',
        summary='Predict disatisfaction for customer by index number',
        description='Loads a JSON file from the examples folder based on the index number entered, if it does not exist, generate it using the GET first'
        )
def predict_from_file(number: int) -> PredictResponse:
    file_path = f'examples/{number}.json'
    if not os.path.exists(file_path):
        return {'error': f'File {file_path} not found'}

    with open(file_path, 'r') as f:
        data = json.load(f)

    prob = predict_single(data)
    return PredictResponse(
        dissatisfaction_probability=prob,
        dissatisfied=bool(prob > 0.136)
    )

port = int(os.getenv("PORT", 9696))

if __name__ == '__main__':
    uvicorn.run('predict:app', host='0.0.0.0', port=port, reload=True)