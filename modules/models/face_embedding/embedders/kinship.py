import os
from .base_insightface_embedder import BaseInsightfaceEmbedder

class Kinship_BB(BaseInsightfaceEmbedder):
    def __init__(self,root=""):
        self.name="KinshipResnet100BB"
        self.model_name = os.path.join(root,"OnnxModels","Embedders","resnet100_bb.onnx") # Use the face recognition model
        self.embedder= self.CreateEmbedder(64,root)


class Kinship_FS(BaseInsightfaceEmbedder):
    def __init__(self,root=""):
        self.name="KinshipResnet100FS"
        self.model_name = os.path.join(root,"OnnxModels","Embedders","resnet100_fs.onnx") # Use the face recognition model
        self.embedder= self.CreateEmbedder(64,root)