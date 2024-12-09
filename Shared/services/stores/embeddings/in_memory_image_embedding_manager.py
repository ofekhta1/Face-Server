import pickle
import faiss
import os
import numpy as np
from typing import Union,List,Any
from Shared.models.stored_embedding import StoredDetectorEmbeddings,FaceEmbedding,StoredEmbeddings
from Shared.services.models.model_loader import ModelLoader
from Shared.models.face_info import FaceInfo
import math
from Shared.models.detector_name import DetectorName
from Shared.models.embedder_name import EmbedderName
from Shared.services.util import norm_path
from .base_image_embedding_manager import BaseImageEmbeddingManager
from Shared.services.logging.console_logger import ConsoleLogger
class InMemoryImageEmbeddingManager(BaseImageEmbeddingManager):
    def __init__(self,root_path:str,model_loader:ModelLoader,logger:ConsoleLogger):
        self.model_loader=model_loader
        self.logger=logger
        self.db_embeddings:dict[str,StoredDetectorEmbeddings]={};
        for detector_name,_ in model_loader.model_registry["detectors"].items():
            PKL_PATH=os.path.join(root_path,"static",detector_name,"embeddings.pkl");
            self.db_embeddings[detector_name]= StoredDetectorEmbeddings({},pkl_path=PKL_PATH)
    
    def get_image_faces(self,filename:str,face_num:int,detector_name:str,embedder_name:str)->list[FaceInfo]:
        if(face_num==-2):
            faces=[FaceInfo(bbox=e.box,landmarks=e.landmarks,quality=e.quality,gender=e.gender,age=e.age) for e in self.db_embeddings[detector_name].embeddings[embedder_name].embeddings if e.name.split('_',2)[-1]==filename];
        else:
            for e in self.db_embeddings[detector_name].embeddings[embedder_name].embeddings:
                if e.name.split('_',2)[-1]==filename and face_num==int(e.name.split('_',2)[-2]):
                    return [FaceInfo(bbox=e.box,landmarks=e.landmarks,quality=e.quality,gender=e.gender,age=e.age)]
        return faces;
    def get_image_embeddings(self,filename:str,detector_name:str,embedder_name:str):
        embeddings=[e.embedding for e in self.db_embeddings[detector_name].embeddings[embedder_name].embeddings if e.name.split('_',2)[-1]==filename];
        return embeddings;
    def get_all_embeddings(self,detector_name:str,embedder_name:str,dedup=True,quality_thresh:float=0)->list[FaceEmbedding]:
        embeddings = [
            e
            for e in self.db_embeddings[detector_name]
            .embeddings[embedder_name]
            .embeddings
            if e.quality>=quality_thresh and (not dedup or e.is_dup == False)
        ]
        return embeddings
    
    def add_embedding_typed(self,embedding:Union[FaceEmbedding,List[FaceEmbedding]],detector_name:str,embedder_name:str):
        self.db_embeddings[detector_name].add_embedding_typed(embedder_name,embedding);

    def add_embedding(self,embedding:np.ndarray[np.float32],name:str,box:list[int],detector_name:str,embedder_name:str):
        self.db_embeddings[detector_name].add_embedding(embedder_name,embedding,name,box);
    
    def remove_embedding_by_index(self,index:int,detector_name:str,embedder_name:str):
        self.db_embeddings[detector_name].remove_embedding_by_index(embedder_name,index);
  
    def get_embedding(self,idx:int,detector_name:str,embedder_name:str)->FaceEmbedding:
        return self.db_embeddings[detector_name].get_embedding(embedder_name,idx);

    def get_index_by_name(self,name:str,detector_name:str,embedder_name:str)->int:
        return self.db_embeddings[detector_name].get_index_by_name(embedder_name,name);
        
    def get_embedding_by_name(self,name:str,detector_name:str,embedder_name:str)->FaceEmbedding:
        return self.db_embeddings[detector_name].get_embedding_by_name(embedder_name,name);

    def train_IVFPQ_index(self,data:StoredEmbeddings):
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
    
    def train_HNSW_index(self,data:StoredEmbeddings):
        #M is the amount of connection of each node(datapoint)
        embeddings=[e.embedding for e in data.embeddings];
        M=int(np.log2(len(embeddings)).round())
        d=512;
        # Define the number of subquantizers (m) and number of bits per subquantizer (nbits)
        data.index = faiss.IndexHNSWFlat( d,M);
        data.index.add(embeddings);
    
    def search(self,q_embeddings:np.ndarray[np.float32],k:int,detector_name:DetectorName,embedder_name:EmbedderName,quality:float=0)->list:
        data=self.db_embeddings[detector_name].embeddings[embedder_name];
        if(len(data.embeddings)==0):
            return []
        # Define the number of clusters (nlist) for the IVFPQ index
        threshold = 25600;
        if len(data.embeddings)>=threshold:
            # Define the number of subquantizers (m) and number of bits per subquantizer (nbits)
            self.train_IVFPQ_index(data);
        else:
            ids=[]
            index = faiss.IndexFlatIP(512);
            if(quality>0):
                data.index=faiss.IndexIDMap(index);
                filtered=[]
                for i in range(len(data.embeddings)):
                    if(data.embeddings[i].quality>quality):
                        ids.append(i)
                        filtered.append(data.embeddings[i].embedding)
                if len(filtered)==0:
                    return [filtered];
                data.index.add_with_ids(np.vstack(filtered).astype(np.float32),np.array(ids,dtype=np.int64))
            else:
                data.index = index;
                data.index.add(np.vstack([e.embedding for e in data.embeddings]).astype(np.float32))

        results=self.find_closest_vector(data,q_embeddings,k);
        return results;
         
    def delete_all(self):
        for detector_name,copy in self.db_embeddings.items():
            self.db_embeddings[detector_name]=StoredDetectorEmbeddings({},pkl_path=copy.PKL_PATH);
            path=norm_path(copy.PKL_PATH)
            if os.path.exists(path):
                os.remove(path);  
    def delete(self,detector_name:str):
        copy=self.db_embeddings[detector_name];
        self.db_embeddings[detector_name]=StoredDetectorEmbeddings({},pkl_path=copy.PKL_PATH);
        path=norm_path(copy.PKL_PATH)
        if os.path.exists(path):
            os.remove(path);
   
    def save(self,detector_name:str=None):
        if detector_name is None:
            for detector_name in self.db_embeddings:
                self.save(detector_name);
        else:
            data=self.db_embeddings[detector_name];
            path=norm_path(data.PKL_PATH)
            with open(path, 'wb') as file:
                pickle.dump(data, file)

    def load(self,detector_name:str):
        data=self.db_embeddings[detector_name];
        path=norm_path(data.PKL_PATH)
        if os.path.exists(path):
            with open(path, 'rb') as file:
                self.db_embeddings[detector_name] = pickle.load(file)

    def find_closest_vector(self,data:StoredEmbeddings,q_vectors:np.ndarray[np.float32],k:int):

        distances,indexes = data.index.search(q_vectors.astype(np.float32), k)
        # return indexes based on distance
        # Create a list of objects
        result = []
        for i in range(len(distances)):
            query_result=[]
            for j in range(len(distances[0])):
                if(indexes[i][j]>-1):
                    face_emb=data.embeddings[indexes[i][j]]
                    obj = {'index': indexes[i][j], 'distance': distances[i][j],"Embedding":face_emb}
                    query_result.append(obj)
            result.append(query_result)
        return result;

    
    def update_metadata(self,name:str,metadata:dict[str,Any],detector_name:DetectorName):
        for embedder_name,embeddings in self.db_embeddings[detector_name].embeddings.items():
            idx=self.get_index_by_name(name,detector_name,embedder_name)
            if(idx==-1):
                self.logger.error(f"No embedding found for {name} {detector_name} {embedder_name}")
                continue

            if "age" in metadata and metadata["age"]>=0:
                embeddings.embeddings[idx].age = metadata["age"]
            
            if "gender" in metadata and metadata["gender"]=="W" or metadata["gender"]=="M":
                embeddings.embeddings[idx].gender = metadata["gender"]
        self.save(detector_name)

