import numpy as np
import cv2
class BaseQualityModel:
    def __init__(self):
        self.name="base"
        
    def get_quality_score(self,img:cv2.Mat,faces:list)->list[np.float32]:
        raise Exception("Get Quality Score Not Implemented")
