import os
import cv2
from .base_media_storage import BaseMediaStorage
from Shared.models.detector_name import DetectorName
import numpy as np
class BaseVideoStorage(BaseMediaStorage):

    ALLOWED_EXTENSIONS = {
        ".mp4",
        ".avi",
        ".mkv",
        ".wmv",
        ".webm",
    }
