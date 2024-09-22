from models.face_info import FaceInfo
from models.detector_name import DetectorName
from . import InMemoryImageEmbeddingManager,MilvusImageEmbeddingManager
from services.models.model_loader import ModelLoader
from services import util
from services.processing.face_aligner import FaceAligner
from services.processing.local.local_embedding_generator import LocalEmbeddingGenerator
import numpy as np

class MetadataManager:
    def __init__(self,emb_manager:InMemoryImageEmbeddingManager|MilvusImageEmbeddingManager,
                 face_aligner:FaceAligner,
                 embedding_generator:LocalEmbeddingGenerator,
                 model_loader:ModelLoader):
        self.model_loader=model_loader
        self.emb_manager=emb_manager
        self.face_aligner=face_aligner
        self.embedding_generator=embedding_generator

    
    def get_image_faces(self, filename: str,face_num:int, detector_name: str,embedder_name:str) -> list[FaceInfo]:
        faces = self.emb_manager.get_image_faces(filename,face_num, detector_name,embedder_name=embedder_name)
        if face_num!=-2:
            # convert landmarks to cropped landmarks
            faces[0].landmarks=util.transform_norm_landmarks(np.array(faces[0].landmarks))

        return faces

    def get_detector_indices(self,filename:str,return_detector:DetectorName):
        generated_embeddings = {}

        embedder_name=next(iter(self.model_loader.model_registry["embedders"]))
        for detector_name in self.model_loader.model_registry["detectors"]:
            embs = self.emb_manager.get_image_embeddings(
                    filename, detector_name, embedder_name
                )
            if len(embs) == 0:
                temp_detector=self.model_loader.load_detector(detector_name)
                temp_embedder=self.model_loader.load_embedder(embedder_name)
                img, faces = self.face_aligner.create_aligned_images(
                    filename, temp_detector
                )
                if img is not None and faces is not None:
                    _, new_embs, _ = self.embedding_generator.generate_all_emb(
                        img, faces, filename, temp_detector, temp_embedder
                    )
                    self.emb_manager.add_embedding_typed(
                            new_embs, detector_name, embedder_name
                        )
                    embs=np.array([f.embedding for f in new_embs])

            generated_embeddings[f"{detector_name}_{embedder_name}"] = embs

        detector_indices = util.get_all_detectors_faces(
            generated_embeddings, return_detector,model_loader=self.model_loader
        )    
        return detector_indices