import sys
import os
import cv2
import Shared.services.util as util
from Shared.services.models.face_embedding.quality.base_quality_model import BaseQualityModel
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))

class FaceSizeQuality(BaseQualityModel):
    def __init__(self,root=""):
        self.name="face_size_quality"
   
    def get_quality_score(self,img:cv2.Mat,faces):
        if not isinstance(faces,list):
            #one face
            faces=[faces]
        quality_list=[util.calculate_quality(img,face) for face in faces]
        return quality_list;