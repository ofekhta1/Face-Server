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
    def load_image( filename: str, detector_name: str=""):
        if(not ImageLoader.allowed_file(filename)):
            return None;
    
        if filename.startswith("aligned_") or filename.startswith("detected_"):
            path = os.path.join(AppPaths.STATIC_FOLDER, detector_name, filename)
        else:
            path = os.path.join(AppPaths.UPLOAD_FOLDER, filename)
        img = cv2.imread(path)
        return img       