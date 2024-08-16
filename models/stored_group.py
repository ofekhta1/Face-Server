from models.stored_embedding import FaceEmbedding
from models.embedder_name import EmbedderName
class StoredGroup:
    def __init__(self,index:dict[str,str],id_groups:dict[str,list[str]],core_faces:dict[int,str]) -> None:
        self.index=index
        self.id_groups=id_groups
        self.core_faces=core_faces


class StoredDetectorGroup:
    def __init__(self,groups:dict[EmbedderName,StoredGroup],PKL_PATH:str) -> None:
        self.groups=groups
        self.PKL_PATH=PKL_PATH
        
