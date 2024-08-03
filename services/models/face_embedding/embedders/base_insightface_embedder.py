import sys
import os
import traceback
import insightface
from .base_embedder_model import BaseEmbedderModel
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))


class BaseInsightfaceEmbedder(BaseEmbedderModel):
    def __init__(self,root=""):
        self.name="base_insightface_emb"
        self.model_name = "" # Use the face recognition model
        self.embedder= self.CreateEmbedder(64,root)

   
    def CreateEmbedder(self,size,root):
        try:
            embedder = insightface.model_zoo.get_model(self.model_name)
            embedder.prepare(ctx_id=1, det_thresh=0.5, det_size=(size, size))
            return embedder
        except Exception as e:
            tb = traceback.format_exc()
            print("Error during model initialization:", e)
            return None
    
   
    def embed(self,img,face):
         return self.embedder.get(img,face)