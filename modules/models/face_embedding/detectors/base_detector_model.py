class BaseDetectorModel:
    def __init__(self):
        self.name="base"
        
    def extract_faces(self,img):
        raise Exception("Extract Faces Not Implemented")
