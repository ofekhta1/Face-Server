import pickle
import os
import sys
from models.embedder_name import EmbedderName
from models.detector_name import DetectorName

sys.path.append(os.path.abspath(".."))
from models.stored_group import StoredDetectorGroup, StoredGroup
from services.models.model_loader import ModelLoader


class ImageGroupRepository:
    def __init__(self, root_path: str, model_loader: ModelLoader):
        self.groups: dict[DetectorName, StoredDetectorGroup] = {}
        for model_name, _ in model_loader.model_registry["detectors"].items():
            PKL_PATH = os.path.join(root_path, "static", model_name, "groups.pkl")
            self.groups[model_name] = StoredDetectorGroup({}, PKL_PATH=PKL_PATH)
            self.load_index(model_name)

    def has_group(self, detector_name, embedder_name):
        return (
            detector_name in self.groups
            and embedder_name in self.groups[detector_name].groups
        )

    def save_index(
        self,
        data: dict[str, str],
        id_groups: dict[str, list[str]],
        core_faces: dict,
        detector_name: str,
        embedder_name: str,
    ):
        self.groups[detector_name].groups[embedder_name] = StoredGroup(
            data, id_groups, core_faces
        )
        self.__internal_save_index(detector_name)

        return id_groups

    def __internal_save_index(self, detector_name: DetectorName):
        with open(self.groups[detector_name].PKL_PATH, "wb") as file:
            pickle.dump(self.groups[detector_name], file)

    def load_index(self, detector_name):
        group = self.groups[detector_name]
        if os.path.exists(group.PKL_PATH):
            with open(group.PKL_PATH, "rb") as file:
                self.groups[detector_name] = pickle.load(file)

    def delete_index(self, detector_name):
        group = self.groups[detector_name]
        group.groups = {}
        if os.path.exists():
            os.remove(group.PKL_PATH)

    def change_group_name(
        self, old_id: str, new_id: str, detector_name: str, embedder_name: str
    ):
        group = self.groups[detector_name].groups[embedder_name]
        faces = group.id_groups[old_id]
        if new_id not in group.id_groups:
            group.id_groups[new_id]=[]
        for face in faces:
            group.index[face] = new_id
            group.id_groups[new_id].append(face)
        del group.id_groups[old_id]
        self.__internal_save_index(detector_name)
        return group.id_groups[new_id]
    
    def get_all_faces(
        self, detector_name: DetectorName, embedder_name: EmbedderName
    ) -> dict[str, str]:
        return self.groups[detector_name].groups[embedder_name].index

    def get_id_groups(self, detector_name: DetectorName, embedder_name: EmbedderName):
        group = self.groups[detector_name].groups[embedder_name]
        return group.id_groups

    def get_thumbnails(
        self, detector_name: DetectorName, embedder_name: EmbedderName
    ) -> dict[str, str]:
        if not self.has_group(detector_name, embedder_name):
            return {}
        group = self.groups[detector_name].groups[embedder_name]
        thumbnails: dict[str, str] = {}
        for id in group.id_groups:
            thumbnails[id] = group.id_groups[id][0]
        return thumbnails

    def get_by_id(
        self, detector_name: DetectorName, embedder_name: EmbedderName, group_id: str
    ) -> list[str]:
        return self.groups[detector_name].groups[embedder_name].id_groups[group_id]

    def assign(
        self,
        img_name: str,
        group_id: str,
        detector_name: DetectorName,
        embedder_name: EmbedderName,
    )->list[str]:
        if not self.has_group(detector_name, embedder_name):
            return False
        group = self.groups[detector_name].groups[embedder_name]
        if img_name in group.index:
            old_group_id = group.index[img_name]
            group.id_groups[old_group_id].remove(img_name)

        group.index[img_name] = group_id
        if group_id in group.id_groups:
            group.id_groups[group_id].append(img_name)
        else:
            group.id_groups[group_id]=[img_name]
        self.__internal_save_index(detector_name)
        return group.id_groups[group_id];