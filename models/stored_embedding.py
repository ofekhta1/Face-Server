from faiss import IndexIVFPQ,IndexHNSWFlat,IndexFlatIP
import numpy as np
from typing import Union,List

class FaceEmbedding:
    def __init__(self,name:str,box:list[int],embedding:np.ndarray,gender:str="",age:int=-1):
        self.name=name
        self.box=box#[x1,y1,x2,y2]
        self.embedding=embedding
        self.gender=gender
        self.age=age



class StoredEmbeddings:
    def __init__(self,embeddings:list[FaceEmbedding],index:IndexIVFPQ | IndexHNSWFlat | IndexFlatIP=None):
        self.embeddings=embeddings
        self.index=index


class StoredDetectorEmbeddings:
    def __init__(self,embeddings:dict[str,StoredEmbeddings],pkl_path:str):
        self.embeddings=embeddings
        self.PKL_PATH=pkl_path


    def remove_embedding_by_name(self,name:str,model_name:str):
        index=self.get_index_by_name(name,model_name);
        self.remove_embedding_by_index(index,model_name);
        

    def remove_embedding_by_index(self,model_name:str,index:int):
        if(index>-1):
            self.embeddings[model_name].embeddings.pop(index);
    def add_embedding_typed(self,model_name:str,embedding:Union[FaceEmbedding,List[FaceEmbedding]]):
        if isinstance(embedding, FaceEmbedding):
            existing=self.get_embedding_by_name(model_name,embedding.name);
            if(not existing or len(existing.embedding)==0):
                if(model_name in self.embeddings):
                    self.embeddings[model_name].embeddings.append(embedding);
                else:
                    self.embeddings[model_name]=StoredEmbeddings([embedding])
        elif isinstance(embedding,list):
            if(model_name in self.embeddings):
                for emb in embedding:
                    existing=self.get_embedding_by_name(model_name,emb.name);
                    if(not existing or len(existing.embedding)==0):
                        self.embeddings[model_name].embeddings.append(emb);
            else:
                self.embeddings[model_name]=StoredEmbeddings(embedding)
    
    def add_embedding(self,model_name:str,embedding:list[float],name:str,box:list[int],**kwargs):
        np_emb=np.array(embedding);
        emb=FaceEmbedding(name,box,np_emb)
        if("gender" in kwargs):
            emb.gender=kwargs["gender"]
        if("age" in kwargs):
            emb.age=kwargs["age"]
        self.add_embedding_typed(model_name,emb);


    def get_embedding(self,model_name:str,idx:int)->FaceEmbedding:
        if(len(self.embeddings[model_name].embeddings)>idx):
            return self.embeddings[model_name].embeddings[idx];
        return None;

    def get_index_by_name(self,model_name:str,name:str)->int:
        if( model_name in self.embeddings):
            for index, embedding in enumerate(self.embeddings[model_name].embeddings):
                if embedding.name==name:
                    return index
        return -1

  
    def get_embedding_by_name(self,model_name:str,name:str)->FaceEmbedding:
        idx=self.get_index_by_name(model_name,name);
        if(idx>-1):
            return self.get_embedding(model_name,idx);
        return None;

