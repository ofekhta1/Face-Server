from services.processing.image_loader import ImageLoader
from services.models.face_embedding.detectors.base_detector_model import BaseDetectorModel
import numpy as np
from services import util

class LocalFaceExtractor:

    def add_quality_to_faces(self,img:np.ndarray,faces:list[dict]):
        if faces is None:
            print("Faces are empty!");
            return;
        faces_copy=faces.copy()
        for i in range(len(faces_copy)):
            quality=util.calculate_quality(img,faces_copy[i])
            faces_copy[i]["quality"]=quality;
        # aligned_0 = best quality 
        sorted_indices = sorted(range(len(faces_copy)), key=lambda x: faces_copy[x]["quality"],reverse=True)
        faces_copy=[faces_copy[i] for i in sorted_indices]
        return faces_copy,sorted_indices;
    
    def extract_faces(self,filename: str, model: BaseDetectorModel):
        img =ImageLoader.load_image(filename,model)
        if img is not None:
            faces= model.extract_faces(img)
            faces=self.add_quality_to_faces(img,faces);
            return img,faces;
        return None,None
    
    def extract_faces(self,img, model: BaseDetectorModel):
        if img is not None:
            faces= model.extract_faces(img)
            faces,_=self.add_quality_to_faces(img,faces);
            return img,faces;
        return img,None