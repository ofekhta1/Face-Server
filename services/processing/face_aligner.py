import os
from insightface.utils.face_align import norm_crop
import cv2
from services.models.face_embedding.detectors.base_detector_model import (
    BaseDetectorModel,
)
from config.app_paths import AppPaths
import numpy as np
from .local.local_face_extractor import LocalFaceExtractor
from models.errors import FaceExtractionError
from services.util import face_path


class FaceAligner:
    def __init__(self, face_extractor: LocalFaceExtractor):
        self.face_extractor = face_extractor

    def align_single_image(
        self,
        face_obj: dict,
        selected_face: int,
        faces_dir:str,
        img: np.ndarray,
        detector_name: str,
        filename: str="",
    ):
        
        landmarks = face_obj["kps"].astype(int)
        aligned_filename = face_path(filename, selected_face)
        aligned_path = os.path.join(
            faces_dir, detector_name, aligned_filename
        )
        aligned_img = norm_crop(img, landmarks, 112, "arcface")
        cv2.imwrite(aligned_path, aligned_img)
        return aligned_filename

    def detect_faces_in_image(
        self, filename: str, model: BaseDetectorModel, images: list
    ):
        img, faces = self.face_extractor.extract_faces(filename, model)
        boxes = []
        if faces:
            for face in faces:
                landmarks = face["kps"].astype(int)
                for point in landmarks:
                    cv2.circle(
                        img,
                        (int(point[0]), int(point[1])),
                        5,
                        (0, 0, 255),
                        -1,
                    )
                box = face["bbox"].astype(int).tolist()
                boxes.append(box)
            detected_filename = "detected_" + filename
            detected_path = os.path.join(
                AppPaths.STATIC_FOLDER, model.name, detected_filename
            )

            cv2.imwrite(detected_path, img)
            images.append(detected_filename)

        else:
            images.append(filename)
        return len(faces), boxes

    def create_aligned_images(
        self, file_path: str,save_file_name:str, detector: BaseDetectorModel,img: cv2.Mat = None,faces_dir:str="") -> tuple[np.typing.NDArray[np.uint8], list] | FaceExtractionError:
        if img is None:
            img, faces = self.face_extractor.extract_faces(
                file_path, detector, "resnet50_quality"
            )
        else:
            img, faces = self.face_extractor.extract_faces(
                img, detector, "resnet50_quality"
            )
        faces_dir=AppPaths.STATIC_FOLDER if not faces_dir else faces_dir
    
        if faces is None or len(faces) == 0:
            if img is None:
                return FaceExtractionError(
                    detector_name=detector.name,
                    reason=f"Image {file_path} could not be loaded!",
                )
            return FaceExtractionError(
                detector_name=detector.name, reason="No Faces Detected!"
            )
        filename=os.path.basename(file_path)
        os.makedirs(os.path.join(faces_dir,detector.name),exist_ok=True)
        for i in range(len(faces)):
            aligned_filename = self.align_single_image(
                faces[i], i,faces_dir, img, detector.name,save_file_name
            )
    
        return img, faces