import os
from .base_triton_embedder import BaseTritonEmbedder
from models.embedder_name import EmbedderName

class Triton_Kinship_BB(BaseTritonEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.bb

class Triton_Kinship_SS(BaseTritonEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.ss

class Triton_Kinship_FS(BaseTritonEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.fs

class Triton_Kinship_FD(BaseTritonEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.fd

class Triton_Kinship_MD(BaseTritonEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.md

class Triton_Kinship_MS(BaseTritonEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.ms

class Triton_Kinship_SIBS(BaseTritonEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.sibs
