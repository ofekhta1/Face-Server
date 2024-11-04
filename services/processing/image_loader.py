from services.models.face_embedding.detectors.base_detector_model import BaseDetectorModel
from config.app_paths import AppPaths
import os
import cv2
class ImageLoader:

    ALLOWED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    @staticmethod
    def allowed_file(filename: str):
        extension = os.path.splitext(filename)[1]
        return extension.lower() in ImageLoader.ALLOWED_EXTENSIONS


    @staticmethod
    def load_image( path: str):
        if(not ImageLoader.allowed_file(path)):
            return None;

        img = cv2.imread(path)
        return img       