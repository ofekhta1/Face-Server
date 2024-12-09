import os
import cv2
from .base_media_storage import BaseMediaStorage
from Shared.models.detector_name import DetectorName
import numpy as np
class BaseImageStorage(BaseMediaStorage):

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
    

    def load_image(self, filename: str, detector_name: DetectorName = "") -> tuple[cv2.typing.MatLike, str]:
        """Load an image from Minio, decode it using OpenCV, and save to a temporary file."""
        image_data,temp_file_path=self.load_media_file(filename,detector_name);
        if image_data is None:
            return None,"";
    
        image_np = np.asarray(image_data, dtype="uint8")
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        return img, temp_file_path

