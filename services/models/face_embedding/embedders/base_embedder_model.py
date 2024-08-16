import numpy as np
import cv2
class BaseEmbedderModel:
    def __init__(self):
        self.name="base"
        
    def embed(self,img:cv2.Mat,faces:list)->np.ndarray:
        raise Exception("Extract Faces Not Implemented")
