from faiss import IndexIVFPQ,IndexHNSWFlat,IndexFlatIP
import numpy as np
class StoredModelEmbeddings:
    def __init__(self,names:list[str],face_counts:dict[str,int],embeddings:np.ndarray,pkl_path:str,index:IndexIVFPQ | IndexHNSWFlat | IndexFlatIP=None):
        self.names=names
        self.embeddings=embeddings
        self.index=index
        self.face_counts=face_counts
        self.PKL_PATH=pkl_path
    def remove_embedding_by_name(self,name,model_name):
        index=self.get_index_by_name(name,model_name);
        self.remove_embedding_by_index(index,model_name);
        

    def remove_embedding_by_index(self,index:int):
        if(index>-1):
            self.names.pop(index);
            self.embeddings=np.delete(self.embeddings,index,axis=0);
    
    def add_embedding(self,embedding:np.ndarray[np.float32],name:str):
        existing=self.get_embedding_by_name(name);
        if(len(existing)==0):
            np_emb=np.array(embedding).reshape(1,-1);
            self.names.append(name);
            self.embeddings=np.vstack(
            (self.embeddings,np_emb));
    def set_face_count(self,filename:str,face_count:int):
        self.face_counts[filename]=face_count;
    def get_name(self,idx:int)->str:
        if(len(self.names)>0):
            return self.names[idx];
        return "";

    def get_embedding(self,idx:int)->np.ndarray[np.float32]:
        if(len(self.embeddings)>0):
            return self.embeddings[idx];
        return [];
    def get_index_by_name(self,name:str)->int:
        try:
            return self.names.index(name);
        except ValueError:
            return -1;

    def get_face_count_by_filename(self,name:str)->int:
        try:
            return self.face_counts[name];
        except ValueError:
            return 0;
    def get_embedding_by_name(self,name:str)->np.ndarray[np.float32]:
        try:
            index=self.names.index(name);
            return self.embeddings[index];
        except ValueError:
            return [];