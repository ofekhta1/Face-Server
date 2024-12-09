from Shared.models.detector_name import DetectorName
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from Shared.models.embedder_name import EmbedderName
import numpy as np
from Shared.models.stored_embedding import FaceEmbedding
from Shared.models.face import Face
import os
from models.file_video_input import FileVideoInput
from services.frame_extractor import FrameExtractor
from Shared.services.logging.console_logger import ConsoleLogger
from Shared.services.stores.media.base_image_storage import BaseImageStorage
from Shared.services.stores.embeddings.base_image_embedding_manager import (
    BaseImageEmbeddingManager,
)
from Shared.services.models.model_loader import ModelLoader


class DataUploader:
    def __init__(
        self,
        frame_extractor: FrameExtractor,
        image_storage: BaseImageStorage,
        emb_manager: BaseImageEmbeddingManager,
        model_loader:ModelLoader,
        postprocess_settings:dict,
        logger: ConsoleLogger,
    ):
        self.frame_extractor = frame_extractor
        self.image_storage = image_storage
        self.main_emb_manager = emb_manager
        self.model_loader=model_loader
        self.min_quality=postprocess_settings["MinQuality"]
        self.n_best_faces=postprocess_settings["NumberOfBestFaces"]
        self.logger=logger

    def upload_processed_input_data(
        self,
        input: FileVideoInput,
    ):
        clustering_embedder_name=EmbedderName.resnet50

        for detector_name in self.model_loader.model_registry["detectors"]:

            face_embeddings: list[FaceEmbedding] = input.emb_manager.get_all_embeddings(
                detector_name, clustering_embedder_name,quality_thresh=self.min_quality
            )

            cluster_alg = DBSCAN(min_samples=2, eps=0.65, metric="precomputed")
            # get all the embeddings of faces with sufficient quality
            embeddings = [e.embedding for e in face_embeddings]
            if len(embeddings) == 0:
                return {}
            # create a similarity matrix that includes the cosine similarity between every 2 images in the embedding data
            similarity_matrix = cosine_similarity(embeddings)
            similarity_matrix = np.clip(similarity_matrix, -1, 1)
            # Apply cluster_alg-model that takes the min sample and max distance as returns the groups according to the required distances and min images in group parameters

            labels = cluster_alg.fit_predict(
                1 - similarity_matrix
            )  # Convert similarity to distance
            face_mapping = {
                face_embeddings[i]: str(labels[i]) for i in range(labels.shape[0])
            }

            value_groups = self.__generate_id_groups(face_mapping)
            filtered_groups = self.filter_top_k_faces(value_groups, self.n_best_faces)
            
            # there is some duplicate behavior since im doing this for each detector
            # during save and upload the duplicate files will override
            # maybe keep track of saved frames from other detectors to reduce duplication and make it more efficient
            frame_paths, cropped_dir = self.frame_extractor.extract_frames_by_identity(
                str(input.path), filtered_groups, input.out_path
            )

            for frame_path in frame_paths:
                self.image_storage.fsave_media_file(
                    frame_path, os.path.basename(frame_path)
                )

            for file in os.listdir(cropped_dir):
                if file.endswith(".png"):
                    self.image_storage.fsave_media_file(
                        os.path.join(cropped_dir, file),
                        file,
                        detector_name=detector_name,
                    )
            valid_fe = [
                inner_dict.name
                for inner_lists in filtered_groups.values()
                for inner_dict in inner_lists
                if inner_lists
            ]

            for embedder_name in self.model_loader.model_registry["embedders"]:
                all_embs=input.emb_manager.get_all_embeddings(detector_name,embedder_name,quality_thresh=self.min_quality)
                filtered_embs=[emb for emb in all_embs if emb.name in valid_fe]
                self.main_emb_manager.add_embedding_typed(
                filtered_embs, detector_name, embedder_name
            )

    def __generate_id_groups(self, data: dict[str, str]):
        id_groups: dict[str, list[str]] = {}
        for face in data:
            group_id = data[face]
            if group_id not in id_groups:
                id_groups[group_id] = []
            id_groups[group_id].append(face)
        return id_groups

    def filter_top_k_faces(
        self, faces_dict: dict[str, list[Face]], k: int
    ) -> dict[str, list[Face]]:
        result = {}
        for key, faces in faces_dict.items():
            # Sort the list of faces by quality in descending order
            sorted_faces = sorted(faces, key=lambda face: face.quality, reverse=True)

            # Keep only the top k faces (or all if there are less than k)
            result[key] = sorted_faces[:k]
        return result
