import pickle
import os
import uvicorn
import pandas as pd
from fastapi import FastAPI
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field

model_path = 'models/best_model.pkl'

with open(model_path, 'rb') as f_in:
    metadata = pickle.load(f_in)

model = metadata['model']
feature_names = metadata['feature_names']

class Customer(BaseModel):
    features: dict = Field(..., description="Feature:value pairs matching model training features")

app = FastAPI(title='Customer dissatisfaction classification')

def predict_single(customer: dict) -> float:
    df = pd.DataFrame([customer], columns=feature_names)
    prob = model.predict_proba(df)[0, 1]
    return float(prob)

@app.post('/predict')
def predict(customer, threshold=0.136):
    prob = predict_single(customer.model_dump())

    return {
        'dissatisfaction_probability': prob,
        'dissatisfied': bool(prob > threshold)
        }

if __name__ == '__main__':
    uvicorn.run('predict:app', host='0.0.0.0', port=9696, reload=True)