from Shared.services.models.model_loader import ModelLoader
from Shared.services.processing.media_loader import MediaLoader
import numpy as np
import os
from Shared.services.stores.embeddings import BaseImageEmbeddingManager
from Shared.services.processing.face_aligner import FaceAligner
import tritonclient.http as httpclient
from .triton_client_handler import TritonClientHandler
from Shared.models.face import Face
from Shared.models.stored_embedding import FaceEmbedding
from .triton_face_extractor import TritonFaceExtractor
from typing import Any
from Shared.models.face_info import FaceInfo
from Shared.models.detector_name import DetectorName


class TritonImageProcessor:

    def __init__(
        self,
        emb_manager: BaseImageEmbeddingManager,
        face_aligner: FaceAligner,
        face_extractor: TritonFaceExtractor,
        model_loader: ModelLoader,
    ):
        self.model_loader = model_loader
        self.emb_manager = emb_manager
        self.face_aligner = face_aligner
        self.face_extractor = face_extractor

    async def process_image(
        self, file_path: str, save_file_name: str, faces_dir: str
    ) -> tuple[dict[str, list[np.ndarray]], list[str]]:
        errors = []
        detector_metadata: dict[DetectorName, dict[str, Any]] = {}
        generated_embeddings: dict[str, list[np.ndarray]] = {}
        img = MediaLoader.load_image(file_path)
        
        input_img_arr = np.array([img])
        input_image = httpclient.InferInput(
            "input_image", input_img_arr.shape, datatype="UINT8"
        )
        input_image.set_data_from_numpy(input_img_arr, binary_data=True)

        response = TritonClientHandler.infer("pipeline", [input_image])
        for detector_name in self.model_loader.model_registry["detectors"]:

            os.makedirs(os.path.join(faces_dir, detector_name), exist_ok=True)

            kpss = response.as_numpy(f"{detector_name}_kps")[0]
            dets = response.as_numpy(f"{detector_name}_dets")[0]
            qualities = response.as_numpy(f"{detector_name}_quality_scores")
            genders=response.as_numpy(f"{detector_name}_genders")
            genders=["M" if gender ==1 else "W" for gender in genders]
            ages=response.as_numpy(f"{detector_name}_ages")
            faces = [
                Face(bbox=det[0:4], kps=kps, det_score=det[4], quality=quality[0],gender=gender,age=age)
                for det, kps, quality,gender,age in zip(dets, kpss, qualities,genders,ages)
            ]
            if len(faces) == 0:
                print(f"No faces detected for detector: {detector_name}")
                continue

            sorted_indices = sorted(
                range(len(faces)), key=lambda x: faces[x]["quality"], reverse=True
            )
            faces = [faces[i] for i in sorted_indices]

            aligned_images = [
                self.face_aligner.align_single_image(
                    face, i, img, detector_name, save_file_name, faces_dir
                )
                for i, face in enumerate(faces)
            ]

            for embedder_name in self.model_loader.model_registry["embedders"]:
                embeddings = response.as_numpy(
                    f"{detector_name}_{embedder_name}_embeddings"
                )
                embeddings = [
                    embeddings[i] for i in sorted_indices
                ]  # sort by quality,so it matches the faces order
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

                # Normalize each row
                normalized_embeddings = embeddings / norms
                generated_embeddings[f"{detector_name}_{embedder_name}"] = (
                    normalized_embeddings
                )

                face_embeddings: list[FaceEmbedding] = []
                metadata: list[FaceInfo] = []

                for i, face in enumerate(faces):
                    # face["embedding"]=normalized_embeddings[i] #no need to store embedding cause it is saved as DTO
                    aligned_image_name = aligned_images[i]
                    bbox = [int(coord) for coord in face["bbox"]]
                    landmarks = [(x[0], x[1]) for x in face["kps"]]
                    quality = face["quality"] if "quality" in face else 1
                    age = face["age"] if "age" in face else -1
                    gender = face["gender"] if "gender" in face else ""
                    fe = FaceEmbedding(
                        aligned_image_name,
                        bbox,
                        normalized_embeddings[i],
                        quality=quality,
                        landmarks=landmarks,
                        age=age,
                        gender=gender,
                    )
                    fi = FaceInfo(
                        landmarks=landmarks,
                        bbox=bbox,
                        quality=quality,
                        age=age,
                        gender=gender
                    )
                    metadata.append(fi)
                    face_embeddings.append(fe)

                query = np.array([f.embedding for f in face_embeddings])
                similar = self.emb_manager.search(
                    query, 1, detector_name, embedder_name
                )
                distance_dup_thresh = 0.95
                for i in range(len(similar)):
                    if len(similar[i]) > 0:
                        closest = similar[i][0]
                        if closest["distance"] > distance_dup_thresh:
                            face_embeddings[i].is_dup = True

                detector_metadata[detector_name] = metadata
                self.emb_manager.add_embedding_typed(
                    face_embeddings, detector_name, embedder_name
                )
        return generated_embeddings, detector_metadata, errors
