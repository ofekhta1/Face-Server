import os
import numpy as np
import sys
from itertools import chain
sys.path.append(os.path.abspath('..'))
from models.stored_embedding import FaceEmbedding
from models.face_info import FaceInfo
from ..models.model_loader import ModelLoader
from pymilvus import MilvusClient,DataType
from typing import Union,List

class MilvusImageEmbeddingManager:
    def __init__(self,url,model_loader:ModelLoader):
        self.client = MilvusClient(
            uri=url
        )
        self.model_loader=model_loader
        for detector_name in model_loader.model_registry["detectors"]:
            for embedder_name in model_loader.model_registry["embedders"]:
                collection_name=self.get_collection_name(detector_name,embedder_name)
                if(not self.client.has_collection(collection_name=collection_name)):
                    schema = MilvusClient.create_schema(
                        auto_id=True,
                        enable_dynamic_field=False,
                        )
                    schema.add_field(field_name="Id", datatype=DataType.INT64, is_primary=True)
                    schema.add_field(field_name="Embedding", datatype=DataType.FLOAT_VECTOR, dim=512)
                    schema.add_field(field_name="Box", datatype=DataType.ARRAY, element_type=DataType.INT32, max_capacity=4)
                    schema.add_field(field_name="Landmarks", datatype=DataType.ARRAY, element_type=DataType.FLOAT, max_capacity=10)
                    schema.add_field(field_name="FileName", datatype=DataType.VARCHAR,max_length=128)
                    schema.add_field(field_name="FaceNum", datatype=DataType.INT16)
                    schema.add_field(field_name="Quality", datatype=DataType.FLOAT)
                    schema.add_field(field_name="Age", datatype=DataType.INT16)
                    schema.add_field(field_name="Gender", datatype=DataType.VARCHAR,max_length=5)
                    schema.add_field(field_name="IsDup", datatype=DataType.BOOL)
                    index_params = self.client.prepare_index_params()
                    index_params.add_index(
                        field_name="Embedding",
                        index_type="FLAT",
                        metric_type="IP",
                        params={ "nlist": 128 }
                    )
                    index_params.add_index(
                        field_name="FileName",
                        index_type="INVERTED",
                        index_name="inverted_FN" # Name of the index to be created
                    )
                    self.client.create_collection(
                        collection_name=collection_name,
                        schema=schema,
                        index_params=index_params
                    )

    def get_collection_name(self,detector_name,embedder_name):
        return f"{detector_name}_{embedder_name}";

    def get_image_faces(self,filename:str,face_num:int,detector_name:str,embedder_name:str)->list[FaceInfo]:
        collection_name=self.get_collection_name(detector_name,embedder_name)
        if face_num==-2:
            query=f"FileName=='{filename}'"
        else:
            query=f"FileName=='{filename}' && FaceNum=={face_num}"
        results=self.client.query(collection_name,query,output_fields=["Box","Landmarks","Quality"])
        landmarks=[]
        found=[]
        for result in results:
            landmarks = [[result["Landmarks"][j], result["Landmarks"][j + 1]] for j in range(0, len(result["Landmarks"]), 2)]
            box=result["Box"]
            quality=result["Quality"]
            found.append(FaceInfo(bbox=box,landmarks=landmarks,quality=quality))
        return found 

    def get_image_embeddings(self,filename:str,detector_name:str,embedder_name:str):
        collection_name=self.get_collection_name(detector_name,embedder_name)
        results=self.client.query(collection_name,f"FileName=='{filename}'",output_fields=["Embedding"])
        embeddings=[r['Embedding'] for r in results]
        return embeddings;#convert to array
    def get_all_embeddings(self,detector_name:str,embedder_name:str,dedup:bool=True,quality_thresh=0):
        collection_name=self.get_collection_name(detector_name,embedder_name)
        results=self.client.query(collection_name,f"Quality>={quality_thresh}",output_fields=["Embedding","FileName","FaceNum","Quality","Box"])
        return [self.__build_face_embedding(r) for r in results];
  
    def __generate_data_from_face_embedding(self,embedding:FaceEmbedding):
        parts=embedding.name.split('_',2);
        filename=parts[-1];
        face_num=int(parts[-2])
        landmarks=list(chain.from_iterable(embedding.landmarks))
        data={
            "Embedding":embedding.embedding,
            "FileName":filename,
            "FaceNum":face_num,
            "Box":embedding.box,
            "Landmarks":landmarks,
            "Quality":embedding.quality,
            "Age":embedding.age,
            "Gender":embedding.gender,
            "IsDup":embedding.is_dup
        }
        return data;
    def add_embedding_typed(self,embedding:Union[FaceEmbedding,List[FaceEmbedding]],detector_name:str,embedder_name:str):
        collection_name=self.get_collection_name(detector_name,embedder_name)
        if isinstance(embedding, FaceEmbedding):
            data:dict=self.__generate_data_from_face_embedding(embedding);
        elif isinstance(embedding,list):
            data:list[dict]=[];
            for emb in embedding:
                data.append(self.__generate_data_from_face_embedding(emb));
                
        res=self.client.insert(collection_name,data)

    def add_embedding(self,embedding:np.ndarray[np.float32],name:str,box:list[int],detector_name:str,embedder_name:str,**kwargs):
        collection_name=self.get_collection_name(detector_name,embedder_name)
        parts=name.split('_',2);
        filename=parts[-1];
        face_num=int(parts[-2])
        data={
            "Embedding":embedding,
            "FileName":filename,
            "FaceNum":face_num,
            "Box":box,
            "Age":embedding.age,
            "Gender":embedding.gender
        }
        for key,value in kwargs.items():
            data[key]=value
        res=self.client.insert(collection_name,data)
            
    def remove_embedding_by_index(self,index:int,detector_name:str,embedder_name:str):
        collection_name=self.get_collection_name(detector_name,embedder_name)
        self.client.delete(collection_name,id=index);
  
    def get_embedding(self,idx:int,detector_name:str,embedder_name:str)->FaceEmbedding:
        collection_name=self.get_collection_name(detector_name,embedder_name)
        result=self.client.get(collection_name,ids=idx,output_fields=["Id","Embedding","Box","FileName","FaceNum"],);
        if(len(result)>0):
            return self.__build_face_embedding(result[0])
        return None;

    def __build_face_embedding(self,data):
        name=f"aligned_{data['FaceNum']}_{data['FileName']}"
        box=data["Box"] if "Box" in data else [];
        embedding=data["Embedding"];
        quality=data["Quality"] if "Quality" in data else 1;
        age=data["Age"] if "Age" in data else -1;
        gender=data["Gender"] if "Gender" in data else "";
        is_dup=data["IsDup"] if "IsDup" in data else False;
        landmarks=[]
        if "Landmarks" in data:
            landmarks = [(data["Landmarks"][i], data["Landmarks"][i + 1]) for i in range(0, len(data["Landmarks"]), 2)]
        return FaceEmbedding(name,box,embedding,quality=quality,landmarks=landmarks,gender=gender,age=age,is_dup=is_dup);

    def get_index_by_name(self,name:str,detector_name:str,embedder_name:str)->int:
        parts=name.split('_',2);
        filename=parts[-1];
        collection_name=self.get_collection_name(detector_name,embedder_name)
        result=self.client.query(collection_name,filter=f"FileName=='{filename}'",output_fields=["Id"]);
        if(len(result)>0):
            return result[0]["Id"]
        return -1;
        
    def get_embedding_by_name(self,name:str,detector_name:str,embedder_name:str)->FaceEmbedding:
        parts=name.split('_',2);
        filename=parts[-1];
        face_num=parts[-2];
        collection_name=self.get_collection_name(detector_name,embedder_name)
        result=self.client.query(collection_name,filter=f"FileName=='{filename}' && FaceNum=={face_num}",output_fields=["Id","Embedding","Box","Quality","FileName","FaceNum"]);
        if(len(result)>0):
            return self.__build_face_embedding(result[0])
        return None;

    def search(self,embedding:np.ndarray[np.float32],k:int,detector_name:str,embedder_name:str,quality:float=0):
        collection_name=self.get_collection_name(detector_name,embedder_name)
        results = self.client.search(
            collection_name=collection_name,
            data=embedding,
            output_fields=["Embedding","FileName","FaceNum","Quality"],
            limit=k, # Max. number of search results to return
            filter=f"Quality >= {quality}",
            search_params={"metric_type": "IP", "params": {}} # Search parameters
        )
        
        return [[{"index":result['id'],'distance':result['distance'],'Embedding':self.__build_face_embedding(result['entity'])} for result in single_query] for single_query in results]
    def delete_all(self):
        for detector_name in self.model_loader.model_registry["detectors"]:
            self.delete(detector_name);
    def delete(self,detector_name:str):
        for embedder_name in self.model_loader.model_registry["embedders"]:
            collection_name=self.get_collection_name(detector_name,embedder_name)
            self.client.drop_collection(collection_name);
    
    def save(self,detector_name:str=None):
        pass;
    def load(self,detector_name:str):
        for embedder_name in self.model_loader.model_registry["embedders"]:
            collection_name=self.get_collection_name(detector_name,embedder_name);
            if(self.client.has_collection(collection_name)):
                self.client.load_collection(collection_name)
