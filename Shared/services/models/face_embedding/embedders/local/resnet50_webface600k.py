import os
from Shared.models.embedder_name import EmbedderName
from .base_insightface_embedder import BaseInsightfaceEmbedder

class Local_ResNet50WebFace600K(BaseInsightfaceEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.resnet50
        self.model_path = os.path.join(root,"OnnxModels","Embedders","w600k_r50.onnx") # Use the face recognition model
        self.embedder= self.CreateEmbedder(64,root)
