import os
from typing import BinaryIO
from Shared.models.detector_name import DetectorName
class BaseMediaStorage:
    ALLOWED_EXTENSIONS = {

    }
    def allowed_file(self,filename: str):
        extension = os.path.splitext(filename)[1]
        return extension.lower() in self.ALLOWED_EXTENSIONS

    def load_media_file(self, filename: str, detector_name: DetectorName = "") -> tuple[bytearray, str]:
        raise NotImplementedError(f"load image not implemented in {self.__class__.__name__}")
    def save_media_file(self, image:BinaryIO, filename: str,file_size:int, detector_name: DetectorName="",invalid:bool=False)->bool:
        raise NotImplementedError(f"save image not implemented in {self.__class__.__name__}") 
    
    def fsave_media_file(self, path:str, filename: str, detector_name: DetectorName="",invalid:bool=False)->bool:
        raise NotImplementedError(f"fsave image not implemented in {self.__class__.__name__}") 
    def init_storage(self,**kwargs):
        pass
    
    def serve_media_file(self,path:str,processed:bool):
        raise NotImplementedError(f"serve image not implemented in {self.__class__.__name__}") 

    def remove_media_file(self,filename:str,detector_name:DetectorName="",invalid:bool=False)->bool:
        raise NotImplementedError(f"Remove image not implemented in {self.__class__.__name__}");

    def file_exists(self,filename:str,detector_name:DetectorName,invalid:bool)->bool:
        raise NotImplementedError(f"File exists not implemented in {self.__class__.__name__}");

    def download(self,filename:str,detector_name:DetectorName,invalid:bool)->str:
        raise NotImplementedError(f"Download not implemented in {self.__class__.__name__}");