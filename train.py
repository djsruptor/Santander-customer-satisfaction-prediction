from src.data import get_data, split_data, remove_correlated_features
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import pickle
import os

os.makedirs("models", exist_ok=True)
model_path = "models/best_model.pkl"


def train_model():
    df = get_data()
    df_reduced, dropped = remove_correlated_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_reduced)

    params = {
    'subsample': 0.6,
    'n_estimators': 350,
    'min_child_weight': 2,
    'max_depth': 7,
    'learning_rate': 0.01,
    'colsample_bytree': 0.7
    }

    model = XGBClassifier(
        objective = 'binary:logistic',
        **params,
        eval_metric='aucpr',
        random_state=666
    )

    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)[:,1]
    roc = roc_auc_score(y_val, y_pred)
    pr = average_precision_score(y_val, y_pred)

    print(f'Validation ROC-AUC: {roc:.3f}, PR-AUC: {pr:.3f}')

    model_metadata = {
        "model": model,
        "feature_names": X_train.columns.tolist()
    }

    with open(model_path, 'wb') as f_out:
        pickle.dump(model_metadata, f_out)
    print ('model saved to %s' % model_path)

    return model

if __name__ == '__main':
    train_model()