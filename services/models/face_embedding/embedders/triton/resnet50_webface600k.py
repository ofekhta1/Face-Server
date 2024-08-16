import os
from models.embedder_name import EmbedderName
from .base_triton_embedder import BaseTritonEmbedder

class Triton_ResNet50WebFace600K(BaseTritonEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.resnet50
        self.input_name="faces"
        self.output_name="embeddings"
