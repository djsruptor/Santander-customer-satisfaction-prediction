# Santander Customer Satisfaction Prediction - XGBoost + FastAPI

## Problem Description

This project predicts customer satisfaction for Santander Bank customers using hundreds of anonymized features. The dataset is part of the Kaggle 2016 Santander Customer Satisfaction competition.

Each record represents a single customer and 369 numeric features. The target variable (`TARGET`) is binary:

`0` = Satisfied
`1` = Dissatisfied

The goal is to build a model that identifies which customers are dissatisfied, enabling Santander to reduce churn, improve customer experience, and optimize retention campaigns. Even small improvements in recall for the minority (dissatisfied) class can translate into significant cost savings and higher lifetime value.

This repository implements the full **CRISP-DM** workflow:

- Business understanding
- Data exploratory analysis to detect missing values, data distribution, and correlations
- Model selection, training, and tuning using different approaches
- Performance validation using AUC metrics
- Deployment through FastAPI service by exposing a `\predict` endpoint for real-time inference
- Reproducible and containerized workflow using `uv` for environment management and Docker for portability

## Data Preparation and EDA

The `download_data.py` script retrieves the dataset using `openml` API and save it as a parquet file. 

EDA and pre-processing steps (see `notebook.ipynb`):
- Confirmed no missing values -> No imputation required
- All columns are numerical -> no categorical encoding required
- Identified highly correlated pairs and reduced features (369 -> 203) to improve efficiency
- Target imbalanced (`0`: 96%, `1`: 4%) visualized with bar chart
- Data was split into train/val/test (60/20/20) using `sklearn.model_selection.train_test_split`

## Model Selection, Training, and Tuning

Model selection and tuning are documented in `notebook.ipynb`, with final training logic in `train.py`. 

Models evaluated:
- `LogisticRegressionCV`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `XGBClassifier`

5-fold cross-validation was implemented using `GridSearchCV` and `RandomizedSearchCV` for parameter optimization.
Results were stored and compared using a consolidated performance table.

> While ROC-AUC provides a global ranking metric, PR-AUC better reflects the model's ability to identify dissatisfied customers — a priority for marketing and customer success teams who act on these predictions. For this reason, both metrics were taken into account when evaluating models performance.

**Best performing model was XGBoost:**
- ROC-AUC: 0.848
- PR-AUC: 0.194

## Performance Validation

Additional validation confirmed model stability and fairness:
- Re-trained on full_train(train+validation) and evaluated on remaining 20%
- Computed confusion matrix and classification report
- Determined optimal F1 threshold for better TPR: **0.136**
- Plotted ROC, PR, Precision, Recall and F1 curves
- Verified no single feature contributed over 40% of total importance

> In this context, optimizing the F1-based threshold and prioritizing a higher TPR is essential. Retaining dissatisfied customers has a disproportionately positive business impact, so accepting a moderate increase in false positives (retaining already-satisfied customers) is a reasonable trade-off for capturing more truly dissatisfied ones.

## Model Deployment

A trained model is served through FastAPI via (`predict.py`).

There are two endpoints:

`GET/` - Returns a JSON sample of feature values for a given index
`POST/predict` - Returns satisfaction prediction based on customer number (user input) and threshold

## Reproducibility

- Dataset source: [OpenML 46859](https://www.openml.org/d/46859)
- Size: ~76k rows x 369 features

To reproduce:

```bash
# 1. Environment setup
pip install uv

uv venv .venv
source .venv/bin/activate

uv sync

# 2. Download data
uv run python scripts/download_data.py

# 3. Train model
uv run python train.py

# 4. Run API
uv run uvicorn predict:app --reload

# 5. Deploy and run Docker container
docker build -t santander-service .
docker run -p 9696:9696 santander-service
``` 

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