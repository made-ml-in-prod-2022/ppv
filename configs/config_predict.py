from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin
from yamldataclassconfig.config import YamlDataClassConfig


@dataclass
class ModelConfig(DataClassJsonMixin):
    store_path : str


@dataclass
class ConfigPredict(YamlDataClassConfig):
    data_path: str = None
    predict_path: str = None
    model: ModelConfig = None
