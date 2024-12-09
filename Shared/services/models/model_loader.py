from .face_embedding.embedders import BaseEmbedderModel
from .face_embedding.detectors import BaseDetectorModel
from .face_embedding.genderage import BaseGenderAgeModel
from .face_embedding.quality.base_quality_model import BaseQualityModel
import logging

logging.basicConfig(level=logging.INFO)

class ModelLoader:
    def __init__(self):
        # Use dictionaries for storing models
        self.instances: dict[str, object] = {}
        self.model_registry = {
            'genderAge': {},
            'detectors': {},
            'embedders': {},
            'quality': {}
        }

    def _load_model(self, model_type: str, model_name: str, root: str = "") -> object:
        """Generic model loading function."""
        if model_name in self.instances:
            return self.instances[model_name]
        if model_name in self.model_registry[model_type]:
            logging.info(f"Loading model: {model_name} of type {model_type}")
            model = self.model_registry[model_type][model_name](root=root)
            self.instances[model_name] = model
            return model
        
        logging.error(f"Model '{model_name}' not found in {model_type} registry!")
        raise ValueError(f"Model '{model_name}' not found in {model_type} registry!")

    def load_embedder(self, model_name: str, root: str = "") -> BaseEmbedderModel:
        return self._load_model("embedders", model_name, root)

    def load_detector(self, model_name: str, root: str = "") -> BaseDetectorModel:
        return self._load_model("detectors", model_name, root)

    def load_genderage(self, model_name: str, root: str = "") -> BaseGenderAgeModel:
        return self._load_model("genderAge", model_name, root)

    def load_quality(self, model_name: str, root: str = "") -> BaseQualityModel:
        return self._load_model("quality", model_name, root)
