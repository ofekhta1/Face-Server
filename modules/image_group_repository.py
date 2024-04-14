import pickle
import os;
import sys
sys.path.append(os.path.abspath('..'))
from models.stored_group import StoredModelGroup 
from .model_loader import ModelLoader

class ImageGroupRepository:
    def __init__(self):
        self.groups:dict[str,StoredModelGroup]={};
        for model_name,_ in ModelLoader.models.items():
            PKL_PATH=os.path.join("static",model_name,"groups.pkl");
            self.groups[model_name]= StoredModelGroup({},PKL_PATH=PKL_PATH)
            self.load_index(model_name);
 
    def train_index(self,data:dict,model_name:str):
        for key,images in data.items():
            for image in images:
                self.groups[model_name].index[image]=str(key)
    def save_index(self,model_name):
        group=self.groups[model_name]
        with open(group.PKL_PATH, 'wb') as file:
            pickle.dump(group.index, file)

    def load_index(self,model_name):
        group=self.groups[model_name]
        if os.path.exists(group.PKL_PATH):
            with open(group.PKL_PATH, 'rb') as file:
                group.index = pickle.load(file)

    def delete_index(self,model_name):
        group=self.groups[model_name]
        group.index={}
        if os.path.exists():
            os.remove(group.PKL_PATH);

    def change_group_name(self, old_id:str, new_id:str,model_name:str):
        group=self.groups[model_name]
        for image_name,group_id in group.index.items():
            if(group.index[image_name]==old_id):
                group.index[image_name]=new_id 
                    
    def get_id_groups(self,model_name:str):
        group=self.groups[model_name]
        id_groups:dict[str,list[str]]={};
        for image_name,group_id in group.index.items():
            if group_id not in id_groups:
                id_groups[group_id] = []
            else:
                id_groups[group_id].append(image_name)
        return id_groups;
