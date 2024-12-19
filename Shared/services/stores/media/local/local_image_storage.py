from ..base_image_storage import BaseImageStorage
from .local_media_storage import LocalMediaStorage
from logging import Logger

class LocalImageStorage(BaseImageStorage,LocalMediaStorage):
    def __init__(self,logger: Logger):
        self.logger = logger
    def init_storage(self, **kwargs) -> None:
        """
        Initialize the image storage directories.

        This function sets the pool directory and processed directory attributes based on the provided keyword arguments.
        If the keyword arguments do not contain the required keys, the function will raise a KeyError.

        Parameters:
        - kwargs (dict): Keyword arguments containing the pool directory and processed directory paths.

        Returns:
        - None: This function does not return any value.
        """
        self.pool_dir = kwargs.get("pool_dir")
        self.processed_dir = kwargs.get("processed_dir")
        