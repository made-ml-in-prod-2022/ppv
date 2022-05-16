"""Module for training model"""

import joblib
from argparse import ArgumentParser


import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train(model_config):
    
    train_df = pd.read_csv('data/training/train.csv')
    test_df = pd.read_csv('data/training/test.csv')

    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                'exang', 'oldpeak', 'slope', 'ca', 'thal']
    target = ['condition']

    X_train = train_df[features].values
    y_train = train_df[target].values.flatten()

    if model_config.name == 'LogisticRegression':
        model = LogisticRegression().fit(X_train, y_train)
    elif model_config.name == 'RandomForest':
        model = RandomForestClassifier().fit(X_train, y_train)
    else:
        raise NotImplementedError

    X_test = test_df[features].values
    y_test = test_df[target].values.flatten()

    accuracy = accuracy_score(y_test, model.predict(X_test))

    print(f'Model {model_config.name} was trained. Accuracy = {accuracy:5.3f}')
    
    joblib.dump(model, model_config.store_path)
    print(f'Model saved to {model_config.store_path}')


if __name__ == '__main__':
    parser = set_parser()

    args = parser.parse_args()

    main(args.model_path)
