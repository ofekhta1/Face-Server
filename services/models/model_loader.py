from . import (BaseDetectorModel,BaseEmbedderModel,EranRetinaFaceDetector,
               BaseGenderAgeModel,MobileNet_CelebA,
               SCRFD10G,ResNet50WebFace600K,
               ResNet100GLint360K,RetinaFace10GF,Kinship_FS,Kinship_BB)
from models.detector_name import DetectorName
from models.embedder_name import EmbedderName

class ModelLoader:
    embedders={EmbedderName.resnet100:ResNet100GLint360K,EmbedderName.resnet50:ResNet50WebFace600K,EmbedderName.bb:Kinship_BB,
               EmbedderName.fs:Kinship_FS}
    detectors={DetectorName.retinaface_antelope: SCRFD10G, DetectorName.retinaface_buffalo:RetinaFace10GF
               ,DetectorName.eran_retinaface:EranRetinaFaceDetector}
    
    genderAge={"MobileNetCeleb0.25_CelebA": MobileNet_CelebA}

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
    def load_detector(model_name,root="")-> BaseDetectorModel:
        if(model_name in ModelLoader.instances):
            return ModelLoader.instances[model_name]
        elif model_name in ModelLoader.detectors:
            model= ModelLoader.detectors[model_name](root=root);
            ModelLoader.instances[model_name]=model
            return model
        #Model Doesnt exist!
        return None 
    @staticmethod
    def load_genderage(model_name,root="")->BaseGenderAgeModel:
        if(model_name in ModelLoader.instances):
            return ModelLoader.instances[model_name]
        elif model_name in ModelLoader.genderAge:
            model= ModelLoader.genderAge[model_name](root=root);
            ModelLoader.instances[model_name]=model
            return model

