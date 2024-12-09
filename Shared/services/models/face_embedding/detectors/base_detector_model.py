class BaseDetectorModel:
    def __init__(self):
        self.name="base"
        
    def extract_faces(self,img)->list[dict]:
        raise Exception(f"Extract Faces Not Implemented For Detector: {self.name}")
