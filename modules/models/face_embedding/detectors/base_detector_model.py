class BaseDetectorModel:
    def __init__(self):
        self.name="base"
        
    def extract_faces(self,img)->list[dict]:
        raise Exception("Extract Faces Not Implemented")
