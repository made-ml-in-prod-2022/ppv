from argparse import ArgumentParser

from configs.config_train import ConfigTrain
from configs.config_predict import ConfigPredict
from src.data_prepare import data_prepare
from src.train import train
from src.predict import predict


def train_pipeline():
    config = ConfigTrain()
    config.load('configs/config_train.yml')

    # preparing data for training
    data_prepare(config.input_data_path, config.training.split)

    # train model
    train(config.training.model)

def predict_pipeline(data_path, predicts_path):
    config = ConfigPredict()
    config.load('configs/config_predict.yml')

    predict(config.model.store_path, data_path, predicts_path)


def get_cli_args():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='train for training; predict for prediction')

    parser_train = subparsers.add_parser('train', help='help for train')

    parser_predict = subparsers.add_parser('predict', help='help for predict')
    
    parser_predict.add_argument('-d', '--dataset', dest='data_path',
                                default='data/training/test.csv', 
                                help='path with data for prediction')

    parser_predict.add_argument('-s', '--save', dest='predicts_path',
                                default='answers.csv', 
                                help='path for predictions')
    
    return parser.parse_args()


def main():
    args = get_cli_args()

    if args.command == 'train':
        train_pipeline()
    elif args.command == 'predict':
        predict_pipeline(args.data_path, args.predicts_path)



if __name__ == '__main__':
    main()
    
    