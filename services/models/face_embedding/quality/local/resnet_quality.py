import sys
import os
import traceback
from onnxruntime import InferenceSession
import cv2
from services.models.face_embedding.quality.base_quality_model import BaseQualityModel
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
from insightface.utils.face_align import norm_crop
import numpy as np

class ResnetQualityEmbedder(BaseQualityModel):
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    def __init__(self,root=""):
        self.name="resnet50_quality"
        self.model_path = os.path.join(root,"OnnxModels","Quality","resnet50_quality.onnx") # Use the face recognition model
        self.session = InferenceSession(self.model_path, providers=ResnetQualityEmbedder.providers)
   
    def get_quality_score(self,img:cv2.Mat,faces)->list[np.float32]:
        if not isinstance(faces,list):
            #one face
            faces=[faces]

        aimgs = [norm_crop(img, landmark=face.kps) for face in faces]
        blob=cv2.dnn.blobFromImages(aimgs,1.0/1.0,(112,112),(0,0,0),swapRB=True)
        blob=(blob/255-0.5)*2
        results=self.session.run(["embeddings","quality_scores"],{"faces":blob})
        quality_list = results[1].ravel().tolist()
        return quality_list;