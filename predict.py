import pickle
import os
import uvicorn
import pandas as pd
from fastapi import FastAPI
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field
from src.data import get_data, json_customer, remove_correlated_features

model_path = 'models/best_model.pkl'

with open(model_path, 'rb') as f_in:
    metadata = pickle.load(f_in)

model = metadata['model']
feature_names = metadata['feature_names']

df = get_data()
df_reduced, dropped = remove_correlated_features(df)

class Customer(BaseModel):
    features: dict = Field(..., description="Feature:value pairs matching model training features")

class PredictResponse(BaseModel):
    dissatisfaction_probability: float
    dissatisfied: bool

app = FastAPI(title='Customer dissatisfaction classification')

def predict_single(customer: Dict[str, float]) -> float:
    df = pd.DataFrame([customer], columns=feature_names)
    result = model.predict_proba(df)[0, 1]
    return float(result)

@app.get('/examples/{index}')
def generate_sample(index: int = 0):
    data = json_customer(df_reduced, index)
    return {'message': f'sample {index} generated', 'sample': data}

@app.post('/predict')
def predict(customer: Customer, threshold: float = 0.136) -> PredictResponse:
    prob = predict_single(customer.model_dump())

    return PredictResponse(
        dissatisfaction_probability = prob,
        dissatisfied = bool(prob > threshold)
    )

if __name__ == '__main__':
    uvicorn.run('predict:app', host='0.0.0.0', port=9696, reload=True)