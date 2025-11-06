import pandas as pd
import openml
from sklearn.model_selection import train_test_split
import numpy as np

openml.config.cache_directory = "data/cache"

def get_data(dataset_id=46859):
    dataset = openml.datasets.get_dataset(dataset_id)
    df, *_ = dataset.get_data()
    return df

def split_data(df, target='TARGET', test_size=0.2, val_size=0.25, random_state=666):
    df_full_train, df_test = train_test_split(df, test_size=test_size, stratify=df[target], random_state=random_state)
    df_train, df_val = train_test_split(df_full_train, test_size=val_size, stratify=df_full_train[target], random_state=random_state)

    X_train, y_train = df_train.drop(columns=["TARGET"]), df_train.TARGET
    X_val, y_val = df_val.drop(columns=["TARGET"]), df_val.TARGET
    X_test, y_test = df_test.drop(columns=["TARGET"]), df_test.TARGET

    return X_train, X_val, X_test, y_train, y_val, y_test

def remove_correlated_features(df, threshold=0.9):
    corr = df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    df_reduced = df.drop(columns=to_drop)

    return df_reduced, to_drop