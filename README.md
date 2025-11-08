# MISSING TO UPDATE README!!!!!!!!!!!!!!!!!
# MISSING TO UPDATE README!!!!!!!!!!!!!!!!!
# MISSING TO UPDATE README!!!!!!!!!!!!!!!!!
# MISSING TO UPDATE README!!!!!!!!!!!!!!!!!
# MISSING TO UPDATE README!!!!!!!!!!!!!!!!!
# MISSING TO UPDATE README!!!!!!!!!!!!!!!!!
# MISSING TO UPDATE README!!!!!!!!!!!!!!!!!
# MISSING TO UPDATE README!!!!!!!!!!!!!!!!!

# Santader Customer Satisfaction Prediction

## Project description
Santander wants to identify dissatisfied customers before they leave. The dataset contains anonymized customer features and a binary target `TARGET` (1 = dissatisfied, 0 = satisfied).


## Approach
1. Data loaded from OpenML via `openml` API
2. Preprocessing and EDA in `notebook.ipynb`
3. Multiple models tested
4. Best model selected by AUC score and saved to `models/model.pkl`
5. Deployed locally as a REST API using FastAPI and Docker

## How to run

### Setup
```bash```
uv sync

### Train model
uv run python train.py

### Run API
uv run uvicorn predict:app --reload

### Docker
docker build -t santander-service
docker run -p 8000:8000 santander-service

## Dataset
- Source: [OpenML 46859](https://www.openml.org/d/46859)
- Size: ~76k rows x 370 features
- Target: `TARGET` (1 = dissatisfied, 0 = satisfied)

## Results


## License
MIT