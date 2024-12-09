import os
from Shared.models.embedder_name import EmbedderName
from .base_insightface_embedder import BaseInsightfaceEmbedder

class Local_ResNet100GLint360K(BaseInsightfaceEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.resnet100
        self.model_path = os.path.join(root,"OnnxModels","Embedders","resnet100_mv1mv3.onnx") # Use the face recognition model
        self.embedder= self.CreateEmbedder(64,root)
