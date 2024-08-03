import pickle
import os;
import sys
from models.embedder_name import EmbedderName
from models.detector_name import DetectorName
sys.path.append(os.path.abspath('..'))
from models.stored_group import StoredDetectorGroup,StoredGroup
from ..models import ModelLoader

class ImageGroupRepository:
    def __init__(self,root_path:str):
        self.groups:dict[DetectorName,StoredDetectorGroup]={};
        for model_name,_ in ModelLoader.detectors.items():
            PKL_PATH=os.path.join(root_path,"static",model_name,"groups.pkl");
            self.groups[model_name]= StoredDetectorGroup({},PKL_PATH=PKL_PATH)
            self.load_index(model_name);
    def has_group(self,detector_name,embedder_name):
         return detector_name in self.groups and embedder_name in self.groups[detector_name].groups
            
    def train_index(self,data:dict,detector_name:str,embedder_name:str):
        self.groups[detector_name].groups[embedder_name]=StoredGroup({})
        for key,images in data.items():
            for image in images:
                self.groups[detector_name].groups[embedder_name].index[image]=str(key)
    def save_index(self,detector_name:str):
        group=self.groups[detector_name]
        with open(group.PKL_PATH, 'wb') as file:
            pickle.dump(group, file)

    def load_index(self,detector_name):
        group=self.groups[detector_name]
        if os.path.exists(group.PKL_PATH):
            with open(group.PKL_PATH, 'rb') as file:
               self.groups[detector_name]= pickle.load(file)

    def delete_index(self,detector_name):
        group=self.groups[detector_name]
        group.groups={}
        if os.path.exists():
            os.remove(group.PKL_PATH);

    def change_group_name(self, old_id:str, new_id:str,detector_name:str,embedder_name:str):
        group=self.groups[detector_name].groups[embedder_name]
        for image_name,_ in group.index.items():
            if(group.index[image_name]==old_id):
                group.index[image_name]=new_id 
                    
    def get_id_groups(self,detector_name:DetectorName,embedder_name:EmbedderName):
        group=self.groups[detector_name].groups[embedder_name]
        id_groups:dict[str,list[str]]={};
        for image_name,group_id in group.index.items():
            if group_id not in id_groups:
                id_groups[group_id] = []
            id_groups[group_id].append(image_name)
        return id_groups;

    def get_thumbnails(self,detector_name:DetectorName,embedder_name:EmbedderName)->dict[str,str]:
        if not self.has_group(detector_name,embedder_name):
            return {};
        group=self.groups[detector_name].groups[embedder_name]
        thumbnails:dict[str,str]={};
        for image_name,group_id in group.index.items():
            if group_id not in thumbnails:
                thumbnails[group_id] = image_name
        return thumbnails;
    def get_by_id(self,detector_name:DetectorName,embedder_name:EmbedderName,group_id:str)->list[str]:
        images=[]
        if detector_name in self.groups and embedder_name in self.groups[detector_name].groups:
            index=self.groups[detector_name].groups[embedder_name].index
            for image in index:
                if index[image]==group_id:
                    images.append(image)

        return images;