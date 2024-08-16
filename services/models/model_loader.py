from .face_embedding.embedders import (
    BaseEmbedderModel
)
from .face_embedding.detectors import (
    BaseDetectorModel
)
from .face_embedding.genderage import BaseGenderAgeModel
class ModelLoader:

    def __init__(self):
        self.genderAge = {}
        self.instances = {}
        self.detectors = {}
        self.embedders = {}

    def load_embedder(self, model_name, root="",is_batch:bool=False) -> BaseEmbedderModel:
        if model_name in self.instances:
            return self.instances[model_name]
        elif model_name in self.embedders:
            model = self.embedders[model_name](root=root)
            self.instances[model_name] = model
            return model
        # Model Doesnt exist!
        return None

    def load_detector(self, model_name, root="", is_batch=False) -> BaseDetectorModel:
        if model_name in self.instances:
            return self.instances[model_name]
        elif model_name in self.detectors:
            model = self.detectors[model_name](root=root)
            self.instances[model_name] = model
            return model
        # Model Doesnt exist!
        return None
    
    def load_genderage(self, model_name, root="") -> BaseGenderAgeModel:
        if model_name in self.instances:
            return self.instances[model_name]
        elif model_name in self.genderAge:
            model = self.genderAge[model_name](root=root)
            self.instances[model_name] = model
            return model
        return None
