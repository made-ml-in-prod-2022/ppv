from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin
# from yamldataclassconfig import create_file_path_field
from yamldataclassconfig.config import YamlDataClassConfig


@dataclass
class SplitConfig(DataClassJsonMixin):
    train_size: float
    random_state : int


@dataclass
class ClfConfig(DataClassJsonMixin):
    name: str
    store_path: str


@dataclass
class TrainingConfig(DataClassJsonMixin):
    split: SplitConfig
    model: ClfConfig


@dataclass
class ConfigTrain(YamlDataClassConfig):
    input_data_path: str = None
    training: TrainingConfig = None



if __name__ == '__main__':
    config = Config()
    config.load()
    print(config)
