import numpy as np
class BaseEmbedderModel:
    def __init__(self):
        self.name="base"
        
    def embed(self,img,face)->np.ndarray:
        raise Exception("Extract Faces Not Implemented")
