# Santander Customer Satisfaction Prediction

## Problem description

From frontline support teams to C-suites, customer satisfaction is a key measure of success. Unhappy customers don't stick around. What's more, unhappy customers rarely voice their dissatisfaction before leaving.

Santander Bank is asking Kagglers to help them identify dissatisfied customers early in their relationship. Doing so would allow Santander to take proactive steps to improve a customer's happiness before it's too late.

In this competition, you'll work with hundreds of anonymized features to predict if a customer is satisfied or dissatisfied with their banking experience.

Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

## Approach
1. Data loaded from OpenML via `openml` API and saved into parquet files.
2. Preprocessing, EDA, model comparison and performance validation in `notebook.ipynb`
3. Best model selected by ROC-AUC and PR-AUC score, saved to `models/model.json`
4. Deployed and containerized locally as a REST API using FastAPI and Docker

## Technical Details
- Models tested: LogisticRegressionCV, DecisionTreeClassifier, RandomForestClassifier, XGBClassifier
- Final model: XGBClassifier
- Preprocessing: Removed highly correlated variables  
- Cross-validation: 5-fold stratified CV 
- Environment: Python 3.12, dependencies in `pyproject.toml`

## How to run
### Setup
```bash
uv sync
uv run python scripts/download_data.py
```
### Train model
```bash
uv run python train.py
```
### Run API
```bash
uv run uvicorn predict:app --reload
```
### Docker
```bash
docker build -t santander-service
docker run -p 8000:8000 santander-service
```
## How to test

When accessing http://localhost:8000/docs, there will be two modules:
- GET: Created so testing users can extract a json file with a customer's data (0 <= index < 1000).
- POST: Users must type an index number to predict the dissatisfaction of that customer.

## Dataset
- Source: [OpenML 46859](https://www.openml.org/d/46859)
- Size: ±76k rows x 360 features
- Target: `TARGET` (1 = dissatisfied, 0 = satisfied)

## Results
Final test ROC-AUC scores: 0.848
Final test PR-AUC scores: 0.194

The ROC-AUC is comparable to results from the competition winner models.
The PR-AUC is relatively low, but expected due to the high imbalance of the data (96%-0, 4%-1)

## Citation
@misc{santander-customer-satisfaction,
    author = {Soraya_Jimenez and Will Cukierski},
    title = {Santander Customer Satisfaction},
    year = {2016},
    howpublished = {\url{https://kaggle.com/competitions/santander-customer-satisfaction}},
    note = {Kaggle}
}

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.