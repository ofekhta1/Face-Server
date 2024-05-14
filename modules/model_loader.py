from .models import SCRFD10G,ResNet50WebFace600K,ResNet100GLint360K,RetinaFace10GF
from .models.embedders.base_embedder_model import BaseEmbedderModel
from .models.detectors.base_detector_model import BaseDetectorModel

class ModelLoader:
    embedders={"ResNet100GLint360K":ResNet100GLint360K,"ResNet50WebFace600K":ResNet50WebFace600K}
    detectors={"SCRFD10G": SCRFD10G, "RetinaFace10GF":RetinaFace10GF}
    # models={"buffalo_l":Buffalo_L,"antelopev2":AntelopeV2}

    instances={}

    @staticmethod
    def load_embedder(model_name,root="")-> BaseEmbedderModel:
        if(model_name in ModelLoader.instances):
            return ModelLoader.instances[model_name]
        elif model_name in ModelLoader.embedders:
            model= ModelLoader.embedders[model_name](root=root);
            ModelLoader.instances[model_name]=model
            return model
        #Model Doesnt exist!
        return None 
    
    @staticmethod
    def load_detector(model_name,root="")-> BaseEmbedderModel:
        if(model_name in ModelLoader.instances):
            return ModelLoader.instances[model_name]
        elif model_name in ModelLoader.detectors:
            model= ModelLoader.detectors[model_name](root=root);
            ModelLoader.instances[model_name]=model
            return model
        #Model Doesnt exist!
        return None 
    
    # @staticmethod
    # def load_model(model_name,root="")->BaseDetectorModel:
    #     if(model_name in ModelLoader.instances):
    #         return ModelLoader.instances[model_name]
    #     elif model_name in ModelLoader.models:
    #         model= ModelLoader.models[model_name](root=root);
    #         ModelLoader.instances[model_name]=model
    #         return model
    #     #Model Doesnt exist!
    #     return None 

