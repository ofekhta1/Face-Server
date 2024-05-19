import pickle
import os;
import sys
sys.path.append(os.path.abspath('..'))
from models.stored_group import StoredDetectorGroup,StoredGroup
from .model_loader import ModelLoader

class ImageGroupRepository:
    def __init__(self,root_path:str):
        self.groups:dict[str,StoredDetectorGroup]={};
        for model_name,_ in ModelLoader.detectors.items():
            PKL_PATH=os.path.join(root_path,"static",model_name,"groups.pkl");
            self.groups[model_name]= StoredDetectorGroup({},PKL_PATH=PKL_PATH)
            self.load_index(model_name);
 
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
                group = pickle.load(file)

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
                    
    def get_id_groups(self,detector_name:str,embedder_name:str):
        group=self.groups[detector_name].groups[embedder_name]
        id_groups:dict[str,list[str]]={};
        for image_name,group_id in group.index.items():
            if group_id not in id_groups:
                id_groups[group_id] = []
            else:
                id_groups[group_id].append(image_name)
        return id_groups;
