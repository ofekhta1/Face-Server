import os
from typing import BinaryIO
from Shared.models.detector_name import DetectorName
import shutil
from ..base_media_storage import BaseMediaStorage
import aiofiles
from logging import Logger
import tempfile

class LocalMediaStorage(BaseMediaStorage):
    def __init__(self,logger: Logger):
        self.logger = logger

    def init_storage(self, **kwargs) -> None:
        self.logger.warning("Creating temporary pool and processing dirs...")
        self.pool_dir=tempfile.mkdtemp(suffix="Face_Pool")
        self.processed_dir=tempfile.mkdtemp(suffix="Face_Processed")

    
    def get_file_path(self, filename: str, detector_name: DetectorName = "", invalid: bool = False) -> str:
        """
        Determine the correct file path based on the filename and other parameters.
        """
        if invalid:
            return os.path.join(self.pool_dir, "no_face", filename)
        elif filename.startswith("aligned_") or filename.startswith("detected_"):
            return os.path.join(self.processed_dir, detector_name, filename)
        else:
            return os.path.join(self.pool_dir, filename)

    def load_media_file(self, filename: str, detector_name: DetectorName = "")->tuple[bytearray,str] :
        if not self.allowed_file(filename):
            self.logger.error(f"Invalid file type {filename}")
            return None,""
        
        file_path = self.get_file_path(filename, detector_name)
        CHUNK_SIZE = 1024 * 1024
        _, file_extension = os.path.splitext(filename)


        try:
            image_data = bytearray()
            with open(file_path, "rb") as file:
                while chunk := file.read(CHUNK_SIZE):
                    image_data.extend(chunk)

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(image_data)

            return image_data,temp_file_path
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return None,""
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None,""

    def save_media_file(self, file: BinaryIO, filename: str, file_size: int, detector_name: DetectorName = "", invalid: bool = False) -> bool:
        save_path = self.get_file_path(filename, detector_name, invalid)

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file, buffer)

        return True

        
    def fsave_media_file(self, path: str, filename: str, detector_name: DetectorName = "", invalid: bool = False) -> bool:
        save_path = self.get_file_path(filename, detector_name, invalid)
        shutil.copy(path, save_path)
        return True
    

    async def serve_media_file(self,path:str,processed:bool):
        directory = self.processed_dir if processed else self.pool_dir
        CHUNK_SIZE = 1024 * 1024
    
        # Resolve and sanitize the path
        file_location = os.path.normpath(os.path.join(directory, path))
        
        # Ensure the path is within the directory
        if not file_location.startswith(directory):
            # Yield an error message if used in FastAPI routes
            yield b"Invalid file path."
            return
        
        # Check if the file exists
        if not os.path.exists(file_location):
            yield b"File not found."
            return
        
        # Stream the file
        async with aiofiles.open(file_location, 'rb') as f:
            while chunk := await f.read(CHUNK_SIZE):
                yield chunk

    def remove_media_file(self, filename: str, detector_name: DetectorName = "", invalid: bool = False) -> bool:
        """
        Remove a media file from the storage.

        Parameters:
        - filename (str): The name of the file to be removed.
        - detector_name (DetectorName, optional): The name of the detector if the image is processed.
        - invalid (bool, optional): Whether the image is invalid (no face detected).

        Returns:
        - bool: True if the file was successfully removed, False otherwise.
        """
        try:
            file_path = self.get_file_path(filename, detector_name, invalid)

            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                self.logger.warning(f"File not found: {file_path}")
            
            return True

        except Exception as e:
            self.logger.error(f"Error removing file {filename}: {str(e)}")
            return False
        
    def file_exists(self, filename: str, detector_name: DetectorName, invalid: bool) -> bool:
        """
        Check if a file exists in the storage.

        This function determines the file path based on the provided parameters
        and checks if the file exists at that location.

        Parameters:
        - filename (str): The name of the file to check.
        - detector_name (DetectorName): The name of the detector if the image is processed.
        - invalid (bool): Whether the image is invalid (no face detected).

        Returns:
        - bool: True if the file exists, False otherwise.
        """
        file_path = self.get_file_path(filename, detector_name, invalid)
        return os.path.exists(file_path)
    
    def download(self, filename, detector_name, invalid)->str|None:
        if self.file_exists(filename, detector_name, invalid):
            _, file_extension = os.path.splitext(filename)
            
            with tempfile.NamedTemporaryFile(delete=False,suffix=file_extension) as temp_file:
                file_path = self.get_file_path(filename, detector_name, invalid)
                shutil.copyfile(file_path, temp_file.name)
            
            return temp_file.name
        
        return None