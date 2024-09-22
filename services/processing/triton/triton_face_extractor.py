from services.processing.image_loader import ImageLoader
from services.models.face_embedding.detectors.base_detector_model import BaseDetectorModel
from services.models.face_embedding.quality import BaseQualityModel
import numpy as np
from services import util
from services.models.model_loader import ModelLoader
class TritonFaceExtractor:
    def __init__(self,model_loader:ModelLoader):
        self.model_loader=model_loader;
    

    def add_quality_to_faces(self, img: np.ndarray, faces: list[dict], model: BaseQualityModel) -> tuple[list[dict], list[int]]:
        """
        This function adds quality scores to the detected faces in an image using a given quality model.
        It sorts the faces based on their quality scores in descending order.

        Parameters:
        img (np.ndarray): The input image in which faces are detected.
        faces (list[dict]): A list of dictionaries, where each dictionary represents a detected face.
        model (BaseQualityModel): The quality model used to calculate the quality scores.

        Returns:
        Tuple[List[dict], List[int]]: A tuple containing two elements:
            - A list of dictionaries representing the detected faces with added quality scores.
            - A list of indices representing the sorted order of faces based on their quality scores.
        """
        if faces is None:
            print("Faces are empty!")
            return

        faces_copy = faces.copy()
        quality_list = model.get_quality_score(img, faces_copy)

        for i, quality in enumerate(quality_list):
            faces_copy[i]["quality"] = quality
        # aligned_0 = best quality 
        sorted_indices = sorted(range(len(faces_copy)), key=lambda x: faces_copy[x]["quality"], reverse=True)
        faces_copy = [faces_copy[i] for i in sorted_indices]

        return faces_copy, sorted_indices
    
    def extract_faces(self,filename: str, model: BaseDetectorModel,quality_type="face_size_quality"):
        img =ImageLoader.load_image(filename,model)
        if img is not None:
            faces= model.extract_faces(img)
            if quality_type=="face_size_quality":
                quality_model=self.model_loader.load_quality(quality_type)
                faces=self.add_quality_to_faces(img,faces,quality_model);
            return img,faces;
        return None,None