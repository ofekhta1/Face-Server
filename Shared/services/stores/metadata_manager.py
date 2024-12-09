from Shared.models.face_info import FaceInfo
from Shared.models.detector_name import DetectorName
from Shared.services.stores.embeddings import BaseImageEmbeddingManager
from Shared.services.models.model_loader import ModelLoader
from Shared.services import util
from Shared.services.processing.local.local_embedding_generator import (
    LocalEmbeddingGenerator,
)
import numpy as np
from Shared.services.stores.media import BaseImageStorage
import tempfile
from Shared.models.errors.base_error import BaseError
import os
from typing import Any


class MetadataManager:
    def __init__(
        self,
        emb_manager: BaseImageEmbeddingManager,
        face_aligner,
        image_storage: BaseImageStorage,
        embedding_generator: LocalEmbeddingGenerator,
        model_loader: ModelLoader,
    ):
        self.model_loader = model_loader
        self.emb_manager = emb_manager
        self.face_aligner = face_aligner
        self.image_storage = image_storage
        self.embedding_generator = embedding_generator

    def get_image_faces(
        self, filename: str, face_num: int, detector_name: str, embedder_name: str
    ) -> list[FaceInfo]:
        faces = self.emb_manager.get_image_faces(
            filename, face_num, detector_name, embedder_name=embedder_name
        )
        if face_num != -2:
            # convert landmarks to cropped landmarks
            faces[0].landmarks = util.transform_norm_landmarks(
                np.array(faces[0].landmarks)
            )

        return faces

    def get_detector_indices(
        self, filename: str, return_detector: DetectorName
    ) -> tuple[dict[str, list[int]], dict[str, list[FaceInfo]]] | BaseError:
        generated_embeddings = {}

        embedder_name = next(iter(self.model_loader.model_registry["embedders"]))
        for detector_name in self.model_loader.model_registry["detectors"]:
            embs = self.emb_manager.get_image_embeddings(
                filename, detector_name, embedder_name
            )
            if len(embs) == 0:
                temp_detector = self.model_loader.load_detector(detector_name)
                temp_embedder = self.model_loader.load_embedder(embedder_name)

                img, file_path = self.image_storage.load_image(filename, detector_name)
                faces_dir = tempfile.TemporaryDirectory(
                    prefix=filename, suffix="_faces"
                )

                det_result = self.face_aligner.create_aligned_images(
                    file_path,
                    save_file_name=filename,
                    img=img,
                    faces_dir=faces_dir.name,
                    detector=temp_detector,
                )
                if isinstance(det_result, BaseError):
                    os.rmdir(faces_dir.name)
                    return det_result
                img, faces = det_result
                result = self.embedding_generator.generate_all_emb(
                    img, faces, filename, temp_detector, temp_embedder
                )
                if isinstance(result, BaseError):
                    os.rmdir(faces_dir.name)
                    return result
                img, new_embs, faces = result

                self.emb_manager.add_embedding_typed(
                    new_embs, detector_name, embedder_name
                )

                embs = np.array([f.embedding for f in new_embs])

            generated_embeddings[f"{detector_name}_{embedder_name}"] = embs

        result = util.get_all_detectors_faces(
            filename,
            generated_embeddings,
            return_detector,
            model_loader=self.model_loader,
            image_metadata=self,
        )
        if isinstance(result, BaseError):
            return result
        else:
            detector_indices, detector_metadata = result
            return detector_indices, detector_metadata

    def update_metadata(
        self,
        image: str,
        face_num: int,
        metadata: dict[str, Any],
        detector_name: DetectorName,
    ):
        face_path = util.face_path(image, face_num)
        self.emb_manager.update_metadata(face_path, metadata, detector_name)
        return True
