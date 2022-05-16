"""Module for make predictions"""

import joblib
from argparse import ArgumentParser

import pandas as pd


def predict(model_path, data_path, results_path):
    """Function to make predictions"""

    df = pd.read_csv(data_path)

    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                'exang', 'oldpeak', 'slope', 'ca', 'thal']

    model = joblib.load(model_path)
    predictions = model.predict(df[features].values)
    pd.DataFrame(predictions).to_csv(results_path, index=False, header=False)
    print(f'Predictions where saved to {results_path}')


def set_parser():
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', dest='data_path',
                        default='data/heart_cleveland_upload.csv', 
                        help='path of data to predict')

    parser.add_argument('-m', '--model', dest='model_path',
                        default='model/model.pth', 
                        help='path to model pkl file')

    parser.add_argument('-r', '--results', dest='results_path',
                        default='data/results.csv', 
                        help='path to results save')

    return parser


if __name__ == '__main__':
    parser = set_parser()

    args = parser.parse_args()

    main(args.model_path, 
         args.data_path, 
         args.results_path)
