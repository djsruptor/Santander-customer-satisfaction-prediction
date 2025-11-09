from src.data import get_data, split_data, remove_correlated_features
from sklearn.metrics import roc_auc_score, average_precision_score
from src.config import REDUCED_FILE, MODEL_FILE
from xgboost import XGBClassifier
import pandas as pd
import joblib
import os
import time

os.makedirs("models", exist_ok=True)

def train_model():
    if os.path.exists(REDUCED_FILE):
        df = pd.read_parquet(REDUCED_FILE)
    else: 
        df_raw = get_data()
        df, dropped = remove_correlated_features(df_raw)
        df.to_parquet(REDUCED_FILE)
   
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    params = {

    }

    model = XGBClassifier(
        objective = 'binary:logistic',
        subsample=0.6,
        n_estimators=350,
        min_child_weight=2,
        max_depth=7,
        learning_rate=0.01,
        colsample_bytree=0.7,
        eval_metric='aucpr',
        random_state=666
    )

    model.fit(
        pd.concat([X_train, X_val]),
        pd.concat([y_train, y_val])
    )

    y_pred = model.predict_proba(X_test)[:,1]
    roc = roc_auc_score(y_test, y_pred)
    pr = average_precision_score(y_test, y_pred)
    print(f'Validation ROC-AUC: {roc:.3f}, PR-AUC: {pr:.3f}')

    joblib.dump(
        {
        "model": model,
        "feature_names": X_train.columns.tolist()
    },
    MODEL_FILE)
    print ('model saved to %s' % MODEL_FILE)

    return model

if __name__ == '__main__':

    start = time.time()
    train_model()
    print(f"Training time: {time.time() - start:.2f}s")