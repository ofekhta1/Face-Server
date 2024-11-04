from typing import BinaryIO
import os
import cv2
from models.detector_name import DetectorName
class BaseImageStorage:

    ALLOWED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }
    def allowed_file(self,filename: str):
        extension = os.path.splitext(filename)[1]
        return extension.lower() in self.ALLOWED_EXTENSIONS

    def load_image(self, filename: str, detector_name: DetectorName = "") -> tuple[cv2.typing.MatLike, str]:
        raise NotImplementedError(f"load image not implemented in {self.__class__.__name__}")
    def save_image(self, image:BinaryIO, filename: str,file_size:int, detector_name: DetectorName="",invalid:bool=False)->bool:
        raise NotImplementedError(f"save image not implemented in {self.__class__.__name__}") 
    
    def fsave_image(self, path:str, filename: str, detector_name: DetectorName="",invalid:bool=False)->bool:
        raise NotImplementedError(f"fsave image not implemented in {self.__class__.__name__}") 
    def init_storage(self,**kwargs):
        pass
    
    def serve_image(self,path:str,processed:bool):
        raise NotImplementedError(f"serve image not implemented in {self.__class__.__name__}") 
