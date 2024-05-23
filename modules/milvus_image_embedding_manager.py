import os
import numpy as np
import sys
sys.path.append(os.path.abspath('..'))
from models.stored_embedding import StoredDetectorEmbeddings,FaceEmbedding,StoredEmbeddings
from .model_loader import ModelLoader
import math
from pymilvus import MilvusClient,DataType
from .util import norm_path

class MilvusImageEmbeddingManager:
    def __init__(self,uri):
        self.client = MilvusClient(
            uri=uri
        )
        
        for detector_name in ModelLoader.detectors:
            for embedder_name in ModelLoader.embedders:
                collection_name=self.get_collection_name(detector_name,embedder_name)
                if(not self.client.has_collection(collection_name=collection_name)):
                    schema = MilvusClient.create_schema(
                        auto_id=True,
                        enable_dynamic_field=False,
                        )
                    schema.add_field(field_name="Id", datatype=DataType.INT64, is_primary=True)
                    schema.add_field(field_name="Embedding", datatype=DataType.FLOAT_VECTOR, dim=512)
                    schema.add_field(field_name="Box", datatype=DataType.ARRAY, element_type=DataType.INT32, max_capacity=4)
                    schema.add_field(field_name="FileName", datatype=DataType.VARCHAR,max_length=128)
                    schema.add_field(field_name="FaceNum", datatype=DataType.INT16)
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

    def get_image_boxes(self,filename:str,detector_name:str,embedder_name:str):
        collection_name=self.get_collection_name(detector_name,embedder_name)
        results=self.client.query(collection_name,f"FileName=='{filename}'",output_fields=["Box"])
        # boxes=[e.box for e in self.db_embeddings[detector_name].embeddings[embedder_name].embeddings if e.name.split('_',2)[-1]==filename];
        return [result['Box'] for result in results] #convert to array

    def get_image_embeddings(self,filename:str,detector_name:str,embedder_name:str):
        collection_name=self.get_collection_name(detector_name,embedder_name)
        results=self.client.query(collection_name,f"FileName=='{filename}'",output_fields=["Embedding"])
        embeddings=[r['Embedding'] for r in results]
        return embeddings;#convert to array
    def get_all_embeddings(self,detector_name:str,embedder_name:str):
        collection_name=self.get_collection_name(detector_name,embedder_name)
        results=self.client.query(collection_name,f"Id > 0",output_fields=["Embedding","FileName","FaceNum"])
        return [self.__build_face_embedding(r) for r in results];

    def add_embedding(self,embedding:np.ndarray[np.float32],name:str,box:list[int],detector_name:str,embedder_name:str):
        collection_name=self.get_collection_name(detector_name,embedder_name)
        parts=name.split('_',2);
        filename=parts[-1];
        face_num=int(parts[-2])
        data={
            "Embedding":embedding,
            "FileName":filename,
            "FaceNum":face_num,
            "Box":box
        }
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
        return FaceEmbedding(name,box,embedding);

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
        result=self.client.query(collection_name,filter=f"FileName=='{filename}' && FaceNum=={face_num}",output_fields=["Id","Embedding","Box","FileName","FaceNum"]);
        if(len(result)>0):
            return self.__build_face_embedding(result[0])
        return None;

    def search(self,embedding:np.ndarray[np.float32],k:int,detector_name:str,embedder_name:str):
        collection_name=self.get_collection_name(detector_name,embedder_name)
        results = self.client.search(
            collection_name=collection_name,
            data=embedding,
            output_fields=["Embedding","FileName","FaceNum"],
            limit=k, # Max. number of search results to return
            search_params={"metric_type": "IP", "params": {}} # Search parameters
        )
        
        return [{"index":result['id'],'distance':result['distance'],'Embedding':self.__build_face_embedding(result['entity'])} for result in results[0]]
    def delete_all(self):
        for detector_name in ModelLoader.detectors:
            self.delete(detector_name);
    def delete(self,detector_name:str):
        for embedder_name in ModelLoader.embedders:
            collection_name=self.get_collection_name(detector_name,embedder_name)
            self.client.drop_collection(collection_name);
    
    def save(self,detector_name:str):
        pass;
    def load(self,detector_name:str):
        for embedder_name in ModelLoader.embedders:
            collection_name=self.get_collection_name(detector_name,embedder_name);
            if(self.client.has_collection(collection_name)):
                self.client.load_collection(collection_name)
