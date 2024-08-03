import os
from .base_insightface_embedder import BaseInsightfaceEmbedder
from models.embedder_name import EmbedderName

class Kinship_BB(BaseInsightfaceEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.bb
        self.model_name = os.path.join(root,"OnnxModels","Embedders","resnet100_bb.onnx") # Use the face recognition model
        self.embedder= self.CreateEmbedder(64,root)


class Kinship_FS(BaseInsightfaceEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.fs
        self.model_name = os.path.join(root,"OnnxModels","Embedders","resnet100_fs.onnx") # Use the face recognition model
        self.embedder= self.CreateEmbedder(64,root)