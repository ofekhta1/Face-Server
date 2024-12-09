from numpy import ndarray,float32
class BaseModel:
    def __init__(self):
        self.name="base"
        
    def extract_faces(self,img):
        raise Exception("Extract Faces Not Implemented")

    def embed(self,img,face)->ndarray[float32]:
        raise Exception("Embedding Faces Not Implemented")