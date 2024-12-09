import os
from insightface.utils.face_align import norm_crop
import cv2
from Shared.services.models.face_embedding.detectors.base_detector_model import (
    BaseDetectorModel,
)
import io
from Shared.services.stores.media.base_image_storage import BaseImageStorage
import numpy as np
from Shared.models.errors import FaceExtractionError
from ..util import face_path


class FaceAligner:
    def __init__(self, face_extractor,image_storage:BaseImageStorage):
        self.face_extractor = face_extractor
        self.image_storage = image_storage

    def align_single_image(
        self,
        face_obj: dict,
        selected_face: int,
        img: np.ndarray,
        detector_name: str,
        filename: str="",
        faces_dir=""
    ):
        
        landmarks = face_obj["kps"].astype(int)
        aligned_filename = face_path(filename, selected_face)
        
        aligned_img = norm_crop(img, landmarks, 112, "arcface")
        if faces_dir is not "":
            aligned_path = os.path.join(
                faces_dir, detector_name, aligned_filename
            )
            cv2.imwrite(aligned_path, aligned_img)
        else:
            _, img_encoded = cv2.imencode('.jpg', aligned_img)
            img_binary = io.BytesIO(img_encoded.tobytes())
            self.image_storage.save_media_file(img_binary,aligned_filename,img.size,detector_name)

        return aligned_filename

    def detect_faces_in_image(
        self, filename: str, model: BaseDetectorModel, images: list
    ):
        img,faces = self.face_extractor.extract_faces(filename, model)
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

            _, img_encoded = cv2.imencode('.jpg', img)
            img_binary = io.BytesIO(img_encoded.tobytes())
            self.image_storage.save_media_file(img_binary,detected_filename,img.size,model.name)
            
            images.append(detected_filename)

        else:
            images.append(filename)
        return len(faces), boxes

    def create_aligned_images(
        self, file_path: str,save_file_name:str, detector: BaseDetectorModel,img: cv2.Mat = None,faces_dir:str="") -> tuple[np.typing.NDArray[np.uint8], list] | FaceExtractionError:
        img, faces = self.face_extractor.extract_faces(
            file_path, detector,img, "resnet50_quality"
        )
    
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
        for i in range(len(faces)):
            aligned_filename = self.align_single_image(
                faces[i], i, img, detector.name,save_file_name,faces_dir
            )
    
        return img, faces