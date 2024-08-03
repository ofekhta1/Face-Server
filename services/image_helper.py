import numpy as np
import os
from models.errors import FaceExtractionError,FaceEmbeddingError
from .stores import in_memory_image_embedding_manager,image_group_repository
from models.errors.base_error import BaseError
from models.similar_image import SimilarImage
from models.face_info import FaceInfo
from models.detector_name import DetectorName
from models.embedder_name import EmbedderName
from . import util;
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import time
from services.models import BaseGenderAgeModel,BaseDetectorModel,BaseEmbedderModel,FamilyClassifier


class ImageHelper:

    # Load model on startup
    def __init__(
        self,
        groups: image_group_repository.ImageGroupRepository,
        emb_manager: in_memory_image_embedding_manager.InMemoryImageEmbeddingManager,
        UPLOAD_FOLDER,
        STATIC_FOLDER,
    ):
        self.UPLOAD_FOLDER = UPLOAD_FOLDER
        self.STATIC_FOLDER = STATIC_FOLDER
        self.groups = groups
        self.emb_manager = emb_manager



    def filter(self, threshold, detector_name, embedder_name):
        manager = self.emb_manager
        errors = []
        embeddings = (
            manager.get_all_embeddings(detector_name,embedder_name)
        )
        original_length = len(embeddings)
        for embedding in embeddings:
            valid = self.get_similar_images(
                embedding.embedding,
                embedding.name.split("_")[-1],
                detector_name,
                embedder_name,
            )
            for image in valid:
                match: str = image["name"]
                _, facenum, filename = match.split("_", 2)
                similarity = util.calculate_similarity(
                    self.emb_manager.get_embedding(
                        image["index"], detector_name, embedder_name
                    ).embedding,
                    embedding.embedding,
                )
                if similarity > threshold:
                    manager.remove_embedding_by_index(
                        image["index"], detector_name, embedder_name
                    )
        filtered_length = len(embeddings)
        return original_length - filtered_length

    def enhance_image(self, filename):
        image_path = os.path.join(self.UPLOAD_FOLDER, filename)
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"The image at {image_path} could not be loaded.")
            return

        # Apply slight Gaussian blur to the image to reduce noise
        blurred = cv2.GaussianBlur(image, (3, 3), 0)

        # Sharpen the image by subtracting the Gaussian blur from the original image
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced
