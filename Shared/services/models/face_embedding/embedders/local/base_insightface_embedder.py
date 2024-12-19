import sys
import os
import traceback
from onnxruntime import InferenceSession
import cv2
from .arcface_onnx import ArcFaceONNX
from Shared.services.models.face_embedding.embedders.base_embedder_model import BaseEmbedderModel
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
import numpy as np

class BaseInsightfaceEmbedder(BaseEmbedderModel):
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    def __init__(self,root=""):
        self.name="base_insightface_emb"
        self.model_path = "" # Use the face recognition model
        self.embedder= self.CreateEmbedder(64,root)
   
    def CreateEmbedder(self,size,root):
        try:
                    
            session = InferenceSession(self.model_path, providers=BaseInsightfaceEmbedder.providers)
            embedder = ArcFaceONNX(model_file=self.model_path, session=session)
            embedder.prepare(ctx_id=1, det_thresh=0.5, det_size=(size, size))

            return embedder
        except Exception as e:
            tb = traceback.format_exc()
            print("Error during model initialization:", e)
            return None
    
   
    def embed(self,img:cv2.Mat,faces):

        if isinstance(faces,list):
            embeddings = self.embedder.get(img,faces)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

            # Normalize each row
            normalized_embeddings = embeddings / norms
            return normalized_embeddings;
        #one face
        embedding= self.embedder.get(img,faces)
        norm = np.linalg.norm(embedding, axis=0, keepdims=True)
        normalized_embedding = embedding/norm
        return normalized_embedding
        
        
    def embed_raw(self,img:cv2.Mat|list[cv2.Mat]):
        embeddings = self.embedder.get_raw(img)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Normalize each row
        normalized_embeddings = embeddings / norms
        return normalized_embeddings;