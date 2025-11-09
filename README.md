# Santander Customer Satisfaction Prediction

## Dataset description

The dataset used for this project is part of the Santander Customer Satisfaction competition ran in Kaggle on 2016. It was extracted using OpenML library and downloaded as a parquet file. 

- Source: [OpenML 46859](https://www.openml.org/d/46859)
- Size: ±76k rows x 360 features
- Target: `TARGET` (1 = dissatisfied, 0 = satisfied)

### Competition description 

From frontline support teams to C-suites, customer satisfaction is a key measure of success. Unhappy customers don't stick around. What's more, unhappy customers rarely voice their dissatisfaction before leaving.

Santander Bank is asking Kagglers to help them identify dissatisfied customers early in their relationship. Doing so would allow Santander to take proactive steps to improve a customer's happiness before it's too late.

In this competition, you'll work with hundreds of anonymized features to predict if a customer is satisfied or dissatisfied with their banking experience.

Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

## Problem description

Santander bank noticed that churning customers do not express their dissatisfaction until it's too late, so they need a classification model that allows them to identify upset customers and take some proactive actions befores they leave.

## Solution proposed

The target variable is divided in two: 0 = Satisfied and 1 = Dissatisfied, which indicates this is a binary classification problem. To obtain a useful model, the methodology CRISP-DM is used as a framework.

## Business understanding

There is a high impact on not effectively detecting dissatisfied customers, so it is worth it to sacrifice a higher rate of false positives if it means a higher rate of true positives. If the target variable is highly imbalanced (as usual on customer satisfaction models), this is a trade-off the business should make.

## Workflow

`notebook.ipynb`

1. Loaded the data from OpenML via `openml` API and saved it into parquet files
2. Performed exploratory data analysis (EDA) to find out what pre-processing actions are required:
    - No missing values -> No need for imputation
    - All columns are numerical, no need to encode categorical variables
    - Found highly correlated columns, so we grouped them in pairs and kept only one variable from each pair to reduce the model's size without affecting its performance
    - Target variable is very imbalanced (0: 96%-1: 4%)
3. Trained and tuned 4 classification models to compare their performances and keep the best one
    - __LogisticRegressionCV:__ Uses a grid of Cs and performns cross-validation to select the best regularization hyper-parameter
    - __DecisionTreeClassifier:__ Combined with GridSearchCV to perform cross-validation and find the best combination of hyperparameters provided in a grid
    - __RandomForestClassifier:__ Combined with GridSearchCV to perform cross-validation and find the best combination of hyperparameters provided in a grid
    - __XGBClassifier:__ Combined with RandomizedSearchCV to perform cross-validation and find the best combination of hyperparameters provided in a grid
4. Created a dataframe to store and compare the performance metrics from all models
    - The competition evaluated model performance using ROC-AUC score, but it has demonstrated to be misleading with highly imbalanced target variables, so I used also the PR-AUC metric to compare the models as well
5. Performed additional validation on the best performing model to confirm if any other adjustments are required:
    - Re-trained on full_train(train+validation) and evaluated on test split to verify if it is overfitting
    - Displayed additional metrics through confusion_matrix and classification_report
    - Set an optimal threshold using F1 to detect a higher TPR
    - Plotted ROC-AUC, PR-AUC, Precision, Recall and F1 metrics for better understanding
    - Evaluated feature importance to make sure no single variable represents over 40% of the weight for the model

`train.py`

6. Generated `reduced.parquet` with the processed data
7. Trained and validated the best performing model
8. Generated a .json file using `joblib` to store the trained model as a separate script

`predict.py`

9. Loaded the model using `joblib` and a DataFrame of 1000 customers from `reduced.parquet`
10. Created a FastAPI app with two modules: 
    - GET: Generates a JSON file with all the parameters from the selected index
    - POST: Predicts whether the selected customer is dissatisfied or not using the previously defined thershold

`Dockerfile`

11. Containerized the virtual environment using `uv`, the trained model, and the `examples/` folder with generated JSON files.

** GridSearchCV is a more precise cross-validation method but on heavy models (like XGB) it can be too slow. RandomizedSearchCV improves the tuning speed at a lower precision cost.

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

## Results
- Final test ROC-AUC scores: 0.848
- Final test PR-AUC scores: 0.194

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
