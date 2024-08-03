from services.processing.local.image_loader import ImageLoader
from services.models import BaseDetectorModel
import numpy as np
from services import util

class LocalFaceExtractor:

    def add_quality_to_faces(self,img:np.ndarray,faces:list[dict]):
        faces_copy=faces.copy()
        for i in range(len(faces_copy)):
            quality=util.calculate_quality(img,faces_copy[i])
            faces_copy[i]["quality"]=quality;
        # aligned_0 = best quality 
        faces_copy.sort(key=lambda x: x["quality"],reverse=True)
    
        return faces_copy;
    
    def extract_faces(self,filename: str, model: BaseDetectorModel):
        img =ImageLoader.load_image(filename,model)
        faces= model.extract_faces(img)
        faces=self.add_quality_to_faces(img,faces);
        return img,faces;
        