import pickle
import os
import sys
from Shared.models.embedder_name import EmbedderName
from Shared.models.detector_name import DetectorName
sys.path.append(os.path.abspath(".."))
from Shared.models.stored_group import StoredDetectorGroup, StoredGroup
from ...models.model_loader import ModelLoader
from ...logging import ConsoleLogger

class BaseClusterRepository:

    async def has_group(self, detector_name, embedder_name):
        raise NotImplementedError(f"has group not implemented in {self.__class__.__name__}");
    async def get_group_id(self,name:str, detector_name:DetectorName, embedder_name:EmbedderName)->str:
        raise NotImplementedError(f"get_group_id not implemented in {self.__class__.__name__}");

    async def save_index(
        self,
        data: dict[str, str],
        id_groups: dict[str, list[str]],
        core_faces: dict,
        detector_name: str,
        embedder_name: str,
    ):
        raise NotImplementedError(f"save_index not implemented in {self.__class__.__name__}");


    async def load_index(self, detector_name):
        raise NotImplementedError(f"load_index not implemented in {self.__class__.__name__}");

    async def delete_index(self, detector_name):
        raise NotImplementedError(f"delete_index not implemented in {self.__class__.__name__}");


    async def change_group_name(
        self, old_id: str, new_id: str, detector_name: str, embedder_name: str
    ):
        raise NotImplementedError(f"save_index not implemented in {self.__class__.__name__}");

    
    async def get_all_faces(
        self, detector_name: DetectorName, embedder_name: EmbedderName
    ) -> dict[str, str]:
        raise NotImplementedError(f"save_index not implemented in {self.__class__.__name__}");

    async def get_id_groups(self, detector_name: DetectorName, embedder_name: EmbedderName):
        raise NotImplementedError(f"save_index not implemented in {self.__class__.__name__}");


    async def get_thumbnails(
        self, detector_name: DetectorName, embedder_name: EmbedderName
    ) -> dict[str, str]:
        raise NotImplementedError(f"save_index not implemented in {self.__class__.__name__}");


    async def get_by_id(
        self, detector_name: DetectorName, embedder_name: EmbedderName, group_id: str
    ) -> list[str]:
        raise NotImplementedError(f"save_index not implemented in {self.__class__.__name__}");

    async def assign(
        self,
        img_name: str,
        group_id: str,
        detector_name: DetectorName,
        embedder_name: EmbedderName,
    )->list[str]:
        raise NotImplementedError(f"save_index not implemented in {self.__class__.__name__}");
