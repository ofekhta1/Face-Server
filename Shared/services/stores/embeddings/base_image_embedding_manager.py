import numpy as np
from Shared.models.detector_name import DetectorName
from Shared.models.embedder_name import EmbedderName
from Shared.models.stored_embedding import FaceEmbedding
from Shared.models.face_info import FaceInfo
from typing import Union,List,Any
from Shared.services.util import face_path
class BaseImageEmbeddingManager:


    def get_image_faces(self,filename:str,face_num:int,detector_name:DetectorName,embedder_name:EmbedderName)->list[FaceInfo]:
        raise NotImplementedError(f"get_image_faces not implemented in {self.__class__.__name__}")
    def get_image_embeddings(self,filename:str,detector_name:DetectorName,embedder_name:EmbedderName):
        raise NotImplementedError(f"get_image_embeddings not implemented in {self.__class__.__name__}")

    def get_all_embeddings(self,detector_name:DetectorName,embedder_name:EmbedderName,dedup:bool=True,quality_thresh=0):
        raise NotImplementedError(f"get_all_embeddings not implemented in {self.__class__.__name__}")


    def add_embedding_typed(self,embedding:Union[FaceEmbedding,List[FaceEmbedding]],detector_name:DetectorName,embedder_name:EmbedderName):
        raise NotImplementedError(f"add_embedding_typed not implemented in {self.__class__.__name__}")

    def add_embedding(self,embedding:np.ndarray[np.float32],name:str,box:list[int],detector_name:DetectorName,embedder_name:EmbedderName,**kwargs):
        raise NotImplementedError(f"add_embedding not implemented in {self.__class__.__name__}")
            
    def remove_embedding_by_index(self,index:int,detector_name:DetectorName,embedder_name:EmbedderName):
        raise NotImplementedError(f"remove_embedding_by_index not implemented in {self.__class__.__name__}")

    def get_embedding(self,idx:int,detector_name:DetectorName,embedder_name:EmbedderName)->FaceEmbedding:
        raise NotImplementedError(f"get_embedding not implemented in {self.__class__.__name__}")

    def get_index_by_name(self,name:str,detector_name:DetectorName,embedder_name:EmbedderName)->int:
        raise NotImplementedError(f"get_index_by_name not implemented in {self.__class__.__name__}")
    def get_embedding_by_name(self,name:str,detector_name:DetectorName,embedder_name:EmbedderName)->FaceEmbedding:
        raise NotImplementedError(f"get_embedding_by_name not implemented in {self.__class__.__name__}")

    def update_metadata(self,name:str,metadata:dict[str,Any],detector_name:DetectorName):
        raise NotImplementedError(f"update_metadata not implemented in {self.__class__.__name__}")
    def search(self,embedding:np.ndarray[np.float32],k:int,detector_name:DetectorName,embedder_name:EmbedderName,quality:float=0):
        raise NotImplementedError(f"search not implemented in {self.__class__.__name__}")
    def delete_all(self):
        raise NotImplementedError(f"delete_all not implemented in {self.__class__.__name__}");

    def delete(self,detector_name:DetectorName):
        raise NotImplementedError(f"delete not implemented in {self.__class__.__name__}");
      
    def save(self,detector_name:DetectorName=None):
        raise NotImplementedError(f"save not implemented in {self.__class__.__name__}");
    def load(self,detector_name:DetectorName):
        raise NotImplementedError(f"load not implemented in {self.__class__.__name__}");