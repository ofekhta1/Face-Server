import os
from models.embedder_name import EmbedderName
from .base_insightface_embedder import BaseInsightfaceEmbedder

class ResNet100GLint360K(BaseInsightfaceEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.resnet100
        self.model_name = os.path.join(root,"OnnxModels","Embedders","glintr100.onnx") # Use the face recognition model
        self.embedder= self.CreateEmbedder(64,root)
