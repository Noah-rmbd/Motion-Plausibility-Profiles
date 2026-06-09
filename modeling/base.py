from abc import ABC, abstractmethod
from pathlib import Path


class ModelPlugin(ABC):
    model_type = None
    capabilities = {}

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def train(self, feature_root, model_dir):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, feature_root, model_dir, prediction_root, datasets):
        raise NotImplementedError

    @classmethod
    def describe(cls):
        return {
            "model_type": cls.model_type,
            "capabilities": cls.capabilities,
        }

    @staticmethod
    def ensure_directory(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

