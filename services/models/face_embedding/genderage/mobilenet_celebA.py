import insightface
import os
from .base_genderage_model import BaseGenderAgeModel

class MobileNet_CelebA(BaseGenderAgeModel):
    def __init__(self,root=""):
        self.name="MobileNetCeleb0.25_CelebA"
        self.model_name = os.path.join(root,"OnnxModels","GenderAge","genderage.onnx") # Use the face recognition model
        self.genderage= self.CreateModel()
    
    def CreateModel(self):
        try:
            embedder = insightface.model_zoo.get_model(self.model_name)
            embedder.prepare(ctx_id=1)
            return embedder
        except Exception as e:
            print("Error during model initialization:", e)
            return None
        #male:1 female:0
    def get_gender_age(self, img, face):
        result=self.genderage.get(img,face);
        if result[0] == 0:
            return "W",result[1];
        return "M",result[1];