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

def predict_pipeline():
    config = ConfigPredict()
    config.load('configs/config_predict.yml')

    predict(config.model.store_path, config.data_path, config.predict_path)


# def set_parser():
#     parser = ArgumentParser()
#     parser.add_argument('-p', '--path', dest='model_path',
#                         default='model/model.pth', 
#                         help='path for model saving')
#     return parser


if __name__ == '__main__':
    predict_pipeline()
