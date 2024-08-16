from .face_embedding.detectors import (
    LocalEranRetinaFaceDetector,
    LocalRetinaFace10GF,
    LocalSCRFD10G,
)
from .face_embedding.embedders import (
    Local_ResNet50WebFace600K,
    Local_ResNet100GLint360K,
    Local_Kinship_FS,
    Local_Kinship_BB,
    Local_Kinship_SS,
    Local_Kinship_MD,
    Local_Kinship_MS,
    Local_Kinship_FD,
    Local_Kinship_SIBS,
)
from .face_embedding.genderage import MobileNet_CelebA
from models.detector_name import DetectorName
from models.embedder_name import EmbedderName
from .model_loader import ModelLoader

class LocalModelLoader(ModelLoader):

    def __init__(self):
        self.embedders = self.__get_local_embedders()
        self.detectors = self.__get_local_detectors()
        self.genderAge = {"MobileNetCeleb0.25_CelebA": MobileNet_CelebA}
        self.instances = {}


    def __get_local_detectors(self):
        return {
            DetectorName.retinaface_buffalo: LocalRetinaFace10GF,
            DetectorName.eran_retinaface: LocalEranRetinaFaceDetector,
        }


    def __get_local_embedders(self):
        return {
            EmbedderName.resnet100: Local_ResNet100GLint360K,
            EmbedderName.resnet50: Local_ResNet50WebFace600K,
            EmbedderName.fs: Local_Kinship_FS,
            EmbedderName.bb: Local_Kinship_BB,
            EmbedderName.ss: Local_Kinship_SS,
            EmbedderName.sibs: Local_Kinship_SIBS,
            EmbedderName.md: Local_Kinship_MD,
            EmbedderName.ms: Local_Kinship_MS,
            EmbedderName.fd: Local_Kinship_FD,
        }






