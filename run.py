from argparse import ArgumentParser
import logging

from configs.config_train import ConfigTrain
from configs.config_predict import ConfigPredict
from ml_project.data_prepare import data_prepare
from ml_project.train import train
from ml_project.predict import predict


def train_pipeline():
    logging.info('Start of train pipeline')

    config = ConfigTrain()
    config.load('configs/config_train.yml')
    
    # preparing data for training
    data_prepare(config.input_data_path, config.training.split)

    # train model
    train(config.training.model)

    logging.info('End of train pipeline')


def predict_pipeline(data_path, predicts_path):
    
    logging.info('Start of predict pipeline')

    config = ConfigPredict()
    config.load('configs/config_predict.yml')

    predict(config.model.store_path, data_path, predicts_path)

    logging.info('End of predict pipeline')


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

    logging.basicConfig(filename='myapp.log', 
                        format='%(asctime)s %(levelname)s:%(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


    if args.command == 'train':
        train_pipeline()
    elif args.command == 'predict':
        predict_pipeline(args.data_path, args.predicts_path)



if __name__ == '__main__':
    main()
    
    
