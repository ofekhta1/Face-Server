import os
import cv2
from typing import BinaryIO
from models.detector_name import DetectorName
import shutil
from .base_image_storage import BaseImageStorage
from config.app_paths import AppPaths
import aiofiles
class LocalImageStorage(BaseImageStorage):
    def __init__(self):
        pass;

    def init_storage(self,**kwargs):
        # Define directories

        self.pool_dir = AppPaths.UPLOAD_FOLDER
        self.processed_dir = AppPaths.STATIC_FOLDER
        
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

    def allowed_file(filename: str):
        extension = os.path.splitext(filename)[1]
        return extension.lower() in LocalImageStorage.ALLOWED_EXTENSIONS


    def load_image(self, filename: str, detector_name: DetectorName = "") -> tuple[cv2.typing.MatLike, str]:
        if(not LocalImageStorage.allowed_file(filename)):
            return None;
    
        if filename.startswith("aligned_") or filename.startswith("detected_"):
            path = os.path.join(self.processed_dir, detector_name, filename)
        else:
            path = os.path.join(self.pool_dir, filename)
        img = cv2.imread(path)
        return img,path
    
    def save_image(self, image:BinaryIO, filename: str,file_size:int, detector_name: DetectorName="",invalid:bool=False)->bool:
        if invalid:
            save_path=os.path.join(self.pool_dir,"no_face",filename);
        elif filename.startswith("aligned_") or filename.startswith("detected_"):
            save_path = os.path.join(self.processed_dir, detector_name, filename)
        else:
            save_path = os.path.join(self.pool_dir, filename)

        with open(save_path,"wb") as buffer:
            shutil.copyfileobj(image,buffer)

        return True
        
    def fsave_image(self, path:str, filename: str, detector_name: DetectorName="",invalid:bool=False)->bool:
        if invalid:
            save_path=os.path.join(self.pool_dir,"no_face",filename);
        elif filename.startswith("aligned_") or filename.startswith("detected_"):
            save_path = os.path.join(self.processed_dir, detector_name, filename)
        else:
            save_path = os.path.join(self.pool_dir, filename)
        
        shutil.move(path,save_path)

        return True
    

    async def serve_image(self,path:str,processed:bool):
        directory= self.processed_dir if processed else self.pool_dir
        CHUNK_SIZE=1024*1024
        file_location = os.path.join(directory, path)
    
        if not os.path.exists(file_location):
            yield None
        async with aiofiles.open(file_location, 'rb') as f:
            while chunk := await f.read(CHUNK_SIZE):
                yield chunk