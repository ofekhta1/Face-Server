import os
from .base_insightface_embedder import BaseInsightfaceEmbedder

class ResNet100GLint360K(BaseInsightfaceEmbedder):
    def __init__(self,root=""):
        self.name="ResNet100GLint360K"
        self.model_name = os.path.join(root,"OnnxModels","Embedders","glintr100.onnx") # Use the face recognition model
        self.embedder= self.CreateEmbedder(64,root)
