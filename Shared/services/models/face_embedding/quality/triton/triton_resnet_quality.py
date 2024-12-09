import sys
import os
import cv2
from Shared.services.processing.triton.triton_client_handler import TritonClientHandler
import tritonclient.http as httpclient
from Shared.services.models.face_embedding.quality.base_quality_model import BaseQualityModel
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
from insightface.utils.face_align import norm_crop
import numpy as np

class TritonResnetQualityEmbedder(BaseQualityModel):
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    def __init__(self,root=""):
        self.name="resnet50_quality"
        self.input_name="faces"
        self.output_name="quality_scores"
    def get_quality_score(self,img:cv2.Mat,faces)->list[np.float32]:
        if not isinstance(faces,list):
            #one face
            faces=[faces]

        aimgs = [norm_crop(img, landmark=face.kps) for face in faces]
        blob=cv2.dnn.blobFromImages(aimgs,1.0/1.0,(112,112),(0,0,0),swapRB=True)
        blob=(blob/255-0.5)*2

        input = httpclient.InferInput(self.input_name, blob.shape, datatype="FP32")
        input.set_data_from_numpy(blob, binary_data=True)
        response=TritonClientHandler.infer(model_name=self.model_name,inputs=[input])
        quality_scores=response.as_numpy(self.output_name);

        return quality_scores;
        
    def get_quality_score_raw(self,img:cv2.Mat|list[cv2.Mat]):
        embeddings = self.embedder.get_raw(img)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Normalize each row
        normalized_embeddings = embeddings / norms
        return normalized_embeddings;