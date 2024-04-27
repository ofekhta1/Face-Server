import pickle
import faiss
import os
import numpy as np
import sys
sys.path.append(os.path.abspath('..'))
from models.stored_embedding import StoredModelEmbeddings,FaceEmbedding
from .model_loader import ModelLoader
import math
from .util import norm_path
class ImageEmbeddingManager:
    def __init__(self,root_path:str):
        
        self.db_embeddings:dict[str,StoredModelEmbeddings]={};
        for model_name,_ in ModelLoader.models.items():
            PKL_PATH=os.path.join(root_path,"static",model_name,"embeddings.pkl");
            self.db_embeddings[model_name]= StoredModelEmbeddings([],pkl_path=PKL_PATH)

    def get_image_boxes(self,filename:str,model_name:str):
        boxes=[e.box for e in self.db_embeddings[model_name].embeddings if e.name.split('_',2)[-1]==filename];
        return boxes;
    def add_embedding(self,embedding:np.ndarray[np.float32],name:str,box:list[int],model_name:str):
        self.db_embeddings[model_name].add_embedding(embedding,name,box);
    
    def remove_embedding_by_index(self,index:int,model_name:str):
        self.db_embeddings[model_name].remove_embedding_by_index(index);
  
    def get_embedding(self,idx:int,model_name:str)->FaceEmbedding:
        return self.db_embeddings[model_name].get_embedding(idx);

    def get_index_by_name(self,name:str,model_name:str)->int:
        return self.db_embeddings[model_name].get_index_by_name(name);
        
    def get_embedding_by_name(self,name:str,model_name:str)->FaceEmbedding:
        return self.db_embeddings[model_name].get_embedding_by_name(name);

    def train_IVFPQ_index(self,data:StoredModelEmbeddings):
        nlist = 100;
        d=512;
        # Define the number of subquantizers (m) and number of bits per subquantizer (nbits)
        m = 16
        embeddings=np.vstack([e.embedding for e in data.embeddings]);
           
        nbits = int(math.floor(np.log2(len(embeddings))))
        quantizer = faiss.IndexFlatIP(d);
        data.index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
        
        data.index.train(embeddings);
        data.index.add(embeddings);
    
    def train_HNSW_index(self,data:StoredModelEmbeddings):
        #M is the amount of connection of each node(datapoint)
        embeddings=[e.embedding for e in data.embeddings];
        M=int(np.log2(len(embeddings)).round())
        d=512;
        # Define the number of subquantizers (m) and number of bits per subquantizer (nbits)
        data.index = faiss.IndexHNSWFlat( d,M);
        data.index.add(embeddings);
    
    def search(self,embedding:np.ndarray[np.float32],k:int,model_name:str):
        data=self.db_embeddings[model_name];
        # Define the number of clusters (nlist) for the IVFPQ index
        threshold = 25600;
        if len(data.embeddings)>=threshold:
            # Define the number of subquantizers (m) and number of bits per subquantizer (nbits)
            self.train_IVFPQ_index(data);
        else:
            data.index = faiss.IndexFlatIP(512);
            data.index.add(np.vstack([e.embedding for e in data.embeddings]))
        return self.find_closest_vector(data,embedding,k);
    def delete_all(self):
        for model_name,copy in self.db_embeddings.items():
            self.db_embeddings[model_name]=StoredModelEmbeddings([],pkl_path=copy.PKL_PATH);
            path=norm_path(copy.PKL_PATH)
            if os.path.exists(path):
                os.remove(path);  
    def delete(self,model_name:str):
        copy=self.db_embeddings[model_name];
        self.db_embeddings[model_name]=StoredModelEmbeddings([],pkl_path=copy.PKL_PATH);
        path=norm_path(copy.PKL_PATH)
        if os.path.exists(path):
            os.remove(path);
    
    def save(self,model_name:str):
        data=self.db_embeddings[model_name];
        path=norm_path(data.PKL_PATH)
        with open(path, 'wb') as file:
            pickle.dump(data, file)

    def load(self,model_name):
        data=self.db_embeddings[model_name];
        path=norm_path(data.PKL_PATH)
        if os.path.exists(path):
            with open(path, 'rb') as file:
                self.db_embeddings[model_name] = pickle.load(file)

    def find_closest_vector(self,data:StoredModelEmbeddings,new_vector:np.ndarray[np.float32],k:int):

        distances,indexes = data.index.search(new_vector, k)
        # return indexes based on distance
        # Create a list of objects
        result = []

        for i in range(len(distances[0])):
            obj = {'index': indexes[0][i], 'distance': distances[0][i]}
            result.append(obj)
        return result;

    