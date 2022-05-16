from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin
from yamldataclassconfig.config import YamlDataClassConfig


@dataclass
class ModelConfig(DataClassJsonMixin):
    store_path : str


@dataclass
class ConfigPredict(YamlDataClassConfig):
    model: ModelConfig = None
