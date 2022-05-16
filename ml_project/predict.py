"""Module for make predictions"""

import joblib
import logging

import pandas as pd


def predict(model_path, data_path, results_path):
    """Function to make predictions"""

    df = pd.read_csv(data_path)

    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                'exang', 'oldpeak', 'slope', 'ca', 'thal']

    model = joblib.load(model_path)
    predictions = model.predict(df[features].values)
    pd.DataFrame(predictions).to_csv(results_path, index=False, header=False)
    logging.info(f'Predictions where saved to {results_path}')
