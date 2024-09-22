from .face_embedding.detectors import (
    TritonRetinaFace10GF,
    LocalEranRetinaFaceDetector,
    TritonEranRetinaFaceDetector
)
from .face_embedding.embedders import (
    Triton_ResNet100GLint360K,
    Triton_ResNet50WebFace600K,
    Triton_Kinship_BB,
    Triton_Kinship_FS,
    Triton_Kinship_SS,
    Triton_Kinship_MD,
    Triton_Kinship_SIBS,
    Triton_Kinship_MS,
    Triton_Kinship_FD,
)
from .face_embedding.genderage import Triton_GenderAge
from models.detector_name import DetectorName
from models.embedder_name import EmbedderName
from services.models.face_embedding.quality import ResnetQualityEmbedder, FaceSizeQuality
from .model_loader import ModelLoader

class TritonModelLoader(ModelLoader):

    def __init__(self):
        super().__init__()  # Initialize the parent class ModelLoader
        self.model_registry['embedders'] = self.__get_triton_embedders()
        self.model_registry['detectors'] = self.__get_triton_detectors()
        self.model_registry['genderAge'] = {"MobileNetCeleb0.25_CelebA": Triton_GenderAge}
        self.model_registry['quality'] = {
            "resnet50_quality": ResnetQualityEmbedder,
            "face_size_quality": FaceSizeQuality
        }

    def __get_triton_detectors(self):
        return {
            DetectorName.retinaface_buffalo: TritonRetinaFace10GF,
            DetectorName.eran_retinaface: TritonEranRetinaFaceDetector,
        }

    def __get_triton_embedders(self):
        return {
            EmbedderName.resnet100: Triton_ResNet100GLint360K,
            EmbedderName.resnet50: Triton_ResNet50WebFace600K,
            EmbedderName.fs: Triton_Kinship_FS,
            EmbedderName.bb: Triton_Kinship_BB,
            EmbedderName.ss: Triton_Kinship_SS,
            EmbedderName.sibs: Triton_Kinship_SIBS,
            EmbedderName.md: Triton_Kinship_MD,
            EmbedderName.ms: Triton_Kinship_MS,
            EmbedderName.fd: Triton_Kinship_FD,
        }
