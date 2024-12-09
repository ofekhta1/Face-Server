import os
from .base_triton_embedder import BaseTritonEmbedder
from Shared.models.embedder_name import EmbedderName



class Triton_Kinship_BB(BaseTritonEmbedder):
    def __init__(self,root=""):
        super(Triton_Kinship_BB, self).__init__();
        self.name=EmbedderName.bb

class Triton_Kinship_SS(BaseTritonEmbedder):
    def __init__(self,root=""):
        super(Triton_Kinship_SS, self).__init__();
        self.name=EmbedderName.ss

class Triton_Kinship_FS(BaseTritonEmbedder):
    def __init__(self,root=""):
        super(Triton_Kinship_FS, self).__init__();
        self.name=EmbedderName.fs

class Triton_Kinship_FD(BaseTritonEmbedder):
    def __init__(self,root=""):
        super(Triton_Kinship_FD, self).__init__();
        self.name=EmbedderName.fd

class Triton_Kinship_MD(BaseTritonEmbedder):
    def __init__(self,root=""):
        super(Triton_Kinship_MD, self).__init__();
        self.name=EmbedderName.md

class Triton_Kinship_MS(BaseTritonEmbedder):
    def __init__(self,root=""):
        super(Triton_Kinship_MS, self).__init__();
        self.name=EmbedderName.ms

class Triton_Kinship_SIBS(BaseTritonEmbedder):
    def __init__(self,root=""):
        super(Triton_Kinship_SIBS, self).__init__();
        self.name=EmbedderName.sibs
