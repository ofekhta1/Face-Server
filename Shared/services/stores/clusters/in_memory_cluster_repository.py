import pickle
import os
import sys
from Shared.models.embedder_name import EmbedderName
from Shared.models.detector_name import DetectorName
sys.path.append(os.path.abspath(".."))
from Shared.models.stored_group import StoredDetectorGroup, StoredGroup
from ...models.model_loader import ModelLoader
from ...logging import ConsoleLogger
from typing import Optional
from .base_cluster_repository import BaseClusterRepository

class InMemoryClusterRepository(BaseClusterRepository):
    def __init__(self, root_path: str, model_loader: ModelLoader, logger: ConsoleLogger):
        self.groups: dict[DetectorName, StoredDetectorGroup] = {}
        self.logger = logger
        for model_name, _ in model_loader.model_registry["detectors"].items():
            pkl_path = os.path.join(root_path, "static", model_name, "groups.pkl")
            self.groups[model_name] = StoredDetectorGroup({}, PKL_PATH=pkl_path)
            self._load_index(model_name)

    async def has_group(self, detector_name: DetectorName, embedder_name: EmbedderName) -> bool:
        return (
            detector_name in self.groups
            and embedder_name in self.groups[detector_name].groups
        )

    async def get_group_id(self, name: str, detector_name: DetectorName, embedder_name: EmbedderName) -> Optional[str]:
        if not await self.has_group(detector_name, embedder_name):
            self.logger.error(f"No group found for {detector_name} {embedder_name}")
            return None
        return self.groups[detector_name].groups[embedder_name].index.get(name)

    async def save_index(
        self,
        data: dict[str, str],
        id_groups: dict[str, list[str]],
        core_faces: dict,
        detector_name: DetectorName,
        embedder_name: EmbedderName,
    ) -> dict[str, list[str]]:
        self.groups[detector_name].groups[embedder_name] = StoredGroup(
            data, id_groups, core_faces
        )
        await self._internal_save_index(detector_name)
        return id_groups

    async def _internal_save_index(self, detector_name: DetectorName):
        """
        Saves the current state of the detector's group data to a pickle file.

        Parameters:
        detector_name (DetectorName): The name of the detector whose group data is to be saved.

        Returns:
        None
        """
        try:
            with open(self.groups[detector_name].PKL_PATH, "wb") as file:
                pickle.dump(self.groups[detector_name], file)
        except Exception as e:
            self.logger.error(f"Failed to save index for {detector_name}: {e}")

    def _load_index(self, detector_name: DetectorName):
        """
        Loads the stored group data for a given detector from a pickle file.

        Parameters:
        detector_name (DetectorName): The name of the detector whose group data is to be loaded.

        Returns:
        None
        """
        group = self.groups[detector_name]
        if os.path.exists(group.PKL_PATH):
            try:
                with open(group.PKL_PATH, "rb") as file:
                    self.groups[detector_name] = pickle.load(file)
            except Exception as e:
                self.logger.error(f"Failed to load index for {detector_name}: {e}")

    async def delete_index(self, detector_name: DetectorName):
        group = self.groups[detector_name]
        group.groups = {}
        if os.path.exists(group.PKL_PATH):
            try:
                os.remove(group.PKL_PATH)
            except Exception as e:
                self.logger.error(f"Failed to delete index for {detector_name}: {e}")

    async def change_group_name(
        self, old_id: str, new_id: str, detector_name: DetectorName, embedder_name: EmbedderName
    ) -> list[str]:
        """
        Changes the name of a group from old_id to new_id for a specified detector and embedder.

        Parameters:
        old_id (str): The current identifier of the group to be renamed.
        new_id (str): The new identifier for the group.
        detector_name (DetectorName): The name of the detector associated with the group.
        embedder_name (EmbedderName): The name of the embedder associated with the group.

        Returns:
        list[str]: A list of face identifiers that are now associated with the new group name.
        """
        group = self.groups[detector_name].groups[embedder_name]
        faces = group.id_groups.pop(old_id, [])
        if new_id not in group.id_groups:
            group.id_groups[new_id] = []
        for face in faces:
            group.index[face] = new_id
            group.id_groups[new_id].append(face)
        await self._internal_save_index(detector_name)
        return group.id_groups[new_id]

    async def get_all_faces(
        self, detector_name: DetectorName, embedder_name: EmbedderName
    ) -> dict[str, str]:
        return self.groups[detector_name].groups[embedder_name].index

    async def get_id_groups(
        self, detector_name: DetectorName, embedder_name: EmbedderName
    ) -> dict[str, list[str]]:
        return self.groups[detector_name].groups[embedder_name].id_groups

    async def get_thumbnails(
        self, detector_name: DetectorName, embedder_name: EmbedderName
    ) -> dict[str, str]:
        if not await self.has_group(detector_name, embedder_name):
            return {}
        group = self.groups[detector_name].groups[embedder_name]
        return {id: faces[0] for id, faces in group.id_groups.items() if faces}

    async def get_by_id(
        self, detector_name: DetectorName, embedder_name: EmbedderName, group_id: str
    ) -> list[str]:
        return self.groups[detector_name].groups[embedder_name].id_groups.get(group_id, [])

    async def assign(
        self,
        img_name: str,
        group_id: str,
        detector_name: DetectorName,
        embedder_name: EmbedderName,
    ) -> Optional[list[str]]:
        if not await self.has_group(detector_name, embedder_name):
            return None
        group = self.groups[detector_name].groups[embedder_name]
        if img_name in group.index:
            old_group_id = group.index[img_name]
            group.id_groups[old_group_id].remove(img_name)
            if not group.id_groups[old_group_id]:
                del group.id_groups[old_group_id]

        group.index[img_name] = group_id
        group.id_groups.setdefault(group_id, []).append(img_name)
        await self._internal_save_index(detector_name)
        return group.id_groups[group_id]