from Shared.models.embedder_name import EmbedderName
from .base_triton_embedder import BaseTritonEmbedder

class Triton_ResNet100GLint360K(BaseTritonEmbedder):
    def __init__(self,root=""):
        self.name=EmbedderName.resnet100
        self.input_name="faces"
        self.output_name="embeddings"
