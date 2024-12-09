from pathlib import Path
from Shared.services.stores.embeddings.base_image_embedding_manager import BaseImageEmbeddingManager
class FileVideoInput:
    def __init__(self,original_path:Path,path:Path,out_path:Path,scale:float,paddings:tuple[float,float]) :
        self.path:Path=path
        self.source_path:Path=original_path
        self.uri="file://"+str(path)
        self.out_path=out_path
        self.bin_name=None
        self.scale=scale
        self.paddings=paddings
        self.emb_manager:BaseImageEmbeddingManager=None