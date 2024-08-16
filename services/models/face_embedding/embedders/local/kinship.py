import os
from .base_insightface_embedder import BaseInsightfaceEmbedder
from models.embedder_name import EmbedderName

class Local_Kinship_BB(BaseInsightfaceEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.bb
        self.model_path = os.path.join(root,"OnnxModels","Embedders","resnet100_bb.onnx") # Use the face recognition model
        self.embedder= self.CreateEmbedder(64,root)


class Local_Kinship_FS(BaseInsightfaceEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.fs
        self.model_path = os.path.join(root,"OnnxModels","Embedders","resnet100_fs.onnx") # Use the face recognition model
        self.embedder= self.CreateEmbedder(64,root)

class Local_Kinship_FD(BaseInsightfaceEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.fd
        self.model_path = os.path.join(root,"OnnxModels","Embedders","resnet100_fd.onnx") # Use the face recognition model
        self.embedder= self.CreateEmbedder(64,root)
class Local_Kinship_MD(BaseInsightfaceEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.md
        self.model_path = os.path.join(root,"OnnxModels","Embedders","resnet100_md.onnx") # Use the face recognition model
        self.embedder= self.CreateEmbedder(64,root)

class Local_Kinship_MS(BaseInsightfaceEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.ms
        self.model_path = os.path.join(root,"OnnxModels","Embedders","resnet100_ms.onnx") # Use the face recognition model
        self.embedder= self.CreateEmbedder(64,root)

class Local_Kinship_SIBS(BaseInsightfaceEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.sibs
        self.model_path = os.path.join(root,"OnnxModels","Embedders","resnet100_sibs.onnx") # Use the face recognition model
        self.embedder= self.CreateEmbedder(64,root)

class Local_Kinship_SS(BaseInsightfaceEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.ss
        self.model_path = os.path.join(root,"OnnxModels","Embedders","resnet100_ss.onnx") # Use the face recognition model
        self.embedder= self.CreateEmbedder(64,root)