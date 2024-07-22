from typing import Optional
from .base_error import BaseError
class FaceEmbeddingError(BaseError):
    embedder_name:str
    
    