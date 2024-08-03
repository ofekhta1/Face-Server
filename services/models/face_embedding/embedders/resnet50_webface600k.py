import os
from models.embedder_name import EmbedderName
from .base_insightface_embedder import BaseInsightfaceEmbedder

class ResNet50WebFace600K(BaseInsightfaceEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.resnet50
        self.model_name = os.path.join(root,"OnnxModels","Embedders","w600k_r50.onnx") # Use the face recognition model
        self.embedder= self.CreateEmbedder(64,root)
