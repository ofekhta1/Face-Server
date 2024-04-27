from faiss import IndexIVFPQ,IndexHNSWFlat,IndexFlatIP
import numpy as np


class FaceEmbedding:
    def __init__(self,name:str,box:list[int],embedding:np.ndarray):
        self.name=name
        self.box=box#[x1,y1,x2,y2]
        self.embedding=embedding

class StoredModelEmbeddings:
    def __init__(self,embeddings:list[FaceEmbedding],pkl_path:str,index:IndexIVFPQ | IndexHNSWFlat | IndexFlatIP=None):
        self.embeddings=embeddings
        self.index=index
        self.PKL_PATH=pkl_path


    def remove_embedding_by_name(self,name,model_name):
        index=self.get_index_by_name(name,model_name);
        self.remove_embedding_by_index(index,model_name);
        

    def remove_embedding_by_index(self,index:int):
        if(index>-1):
            self.embeddings.pop(index);
    
    def add_embedding(self,embedding:list[float],name:str,box:list[int]):
        existing=self.get_embedding_by_name(name);
        if(not existing or len(existing.embedding)==0):
            np_emb=np.array(embedding);
            self.embeddings.append(FaceEmbedding(name,box,np_emb));

    def get_embedding(self,idx:int)->FaceEmbedding:
        if(len(self.embeddings)>idx):
            return self.embeddings[idx];
        return None;

    def get_index_by_name(self,name:str)->int:
        for index, embedding in enumerate(self.embeddings):
            if embedding.name==name:
                return index
        return -1

  
    def get_embedding_by_name(self,name:str)->FaceEmbedding:
        idx=self.get_index_by_name(name);
        if(idx>-1):
            return self.get_embedding(idx);
        return None;

