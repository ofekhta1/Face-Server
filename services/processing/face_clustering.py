from models.detector_name import DetectorName;
from models.embedder_name import EmbedderName;
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import numpy as np
from services.stores.image_group_repository import ImageGroupRepository
from services.stores import ImageGroupRepository,InMemoryImageEmbeddingManager,MilvusImageEmbeddingManager
from services.models.family.family_classifier import FamilyClassifier
class FaceClustering:
    def __init__(self,emb_manager:InMemoryImageEmbeddingManager|MilvusImageEmbeddingManager,
                 groups:ImageGroupRepository):
        self.emb_manager=emb_manager
        self.groups=groups
    def cluster_images(
        self, max_distance:float, min_samples:int, detector_name:DetectorName, embedder_name:EmbedderName,quality_thresh:float=0,
        retrain=False
    ) -> dict[int, list[str]]:
        face_embeddings =self.emb_manager.get_all_embeddings(detector_name,embedder_name,quality_thresh=quality_thresh)
        dbscan = DBSCAN(eps=max_distance, min_samples=min_samples, metric="precomputed")
        
        # if(not retrain):
        #     existing=self.groups.get_all_faces(detector_name,embedder_name);
        #     #get non clustered faces and embeddings
        #     non_clustered_faces=[]
        #     non_clustered_embeddings=[]
        #     for face_emb in face_embeddings:
        #         if face_emb.name not in existing or existing[face_emb.name]=="-1":
        #             non_clustered_faces.append(face_emb.name)
        #             non_clustered_embeddings.append(face_emb.embedding)

        #     #create clusters of currently non clustered stuff
        #     similarity_matrix = cosine_similarity(non_clustered_embeddings)
            
        #     similarity_matrix = np.clip(similarity_matrix, -1, 1)
            
        #     labels = dbscan.fit_predict(1 - similarity_matrix)  # Convert similarity to distance

        #     id_groups:dict[str,list[str]]={};
        #     for i in range(len(labels)):
        #         group_id=labels[i]
        #         face=non_clustered_faces[i];
        #         if group_id not in id_groups:
        #             id_groups[group_id] = []
        #         id_groups[group_id].append(face)

        #     existing_id_groups=self.groups.get_id_groups(detector_name,embedder_name);
        #     #merge clusters or keep new ones if no merge candidate found
        #     existing_id_groups["-1"]=id_groups[-1]
        #     for j in range(0,len(id_groups)-1):
        #         group_to_merge=id_groups[j];
        #         for k in existing_id_groups:
        #             pass
        #get all the embeddings of faces with sufficient quality
        embeddings=[e.embedding for e in face_embeddings]
        if len(embeddings) == 0:
            return {}
        #create a similarity matrix that includes the cosine similarity between every 2 images in the embedding data
        similarity_matrix = cosine_similarity(embeddings)
        similarity_matrix = np.clip(similarity_matrix, -1, 1)
        # Apply DBSCAN-model that takes the min sample and max distance as returns the groups according to the required distances and min images in group parameters

        labels = dbscan.fit_predict(1 - similarity_matrix)  # Convert similarity to distance
        face_mapping={
            face_embeddings[i].name:str(labels[i])
            for i in range(labels.shape[0])
        }
        core_faces={}
        for index in dbscan.core_sample_indices_:
            group_id=face_mapping[face_embeddings[index].name]
            core_faces[index]=group_id            
             
        value_groups=self.__generate_id_groups(face_mapping)

        if not retrain and self.groups.has_group(detector_name,embedder_name):
            existing_index =self.groups.get_all_faces(detector_name,embedder_name)
            for id in value_groups:
                seen={}
                for face in value_groups[id]:
                    if face in existing_index:
                        if existing_index[face] in seen:
                            seen[existing_index[face]]+=1
                        else:
                            seen[existing_index[face]]=1
                most_common_id=max(seen,key=seen.get);
                if(seen[most_common_id]>=min_samples):
                    # set all values in cluster to most_common_id
                    for face in value_groups[id]:
                        face_mapping[face]=most_common_id

            value_groups=self.__generate_id_groups(face_mapping);
        self.groups.save_index(face_mapping,value_groups,core_faces,detector_name,embedder_name)

        return value_groups;

    def __generate_id_groups(self,data:dict[str,str]):
        id_groups:dict[str,list[str]]={};
        for face in data:
            group_id=data[face]
            if group_id not in id_groups:
                id_groups[group_id] = []
            id_groups[group_id].append(face)
        return id_groups;

    def cluster_images_family(self, max_distance, min_samples, detector_name:str, embedder_name:str,classifier:FamilyClassifier) -> dict[int, list[str]]:
        embeddings=[]
        embeddings=self.emb_manager.get_all_embeddings(detector_name=detector_name,embedder_name=embedder_name);
        Genders = []
        if len(embeddings) == 0:
            return {}
        similarity_matrix = cosine_similarity([e.embedding for e in embeddings])
        Genders=[e.gender for e in embeddings]
        is_same_family = classifier.predict_batch(similarity_matrix, Genders)
        distance_matrix = 1 - is_same_family
        dbscan = DBSCAN(eps=max_distance, min_samples=min_samples, metric='precomputed')
        labels = dbscan.fit_predict(distance_matrix)
        value_groups = {}
        for label in np.unique(labels):
        #  if label != -1:  # Exclude noise points
            value_groups[int(label)] = [embeddings[i].name for i in range(len(labels)) if labels[i] == label]

        return value_groups
    

    def compare_kinship_clusters(self,cluster_id_1,cluster_id_2,
                                 detector_name:DetectorName,embedder_name:EmbedderName,kinship_embedder_name:EmbedderName):
        images1=self.groups.get_by_id(detector_name,embedder_name,cluster_id_1);
        images2=self.groups.get_by_id(detector_name,embedder_name,cluster_id_2);
        embeddings1=[]; 
        embeddings2=[];
        for image in images1:
            emb=self.emb_manager.get_embedding_by_name(image,detector_name,kinship_embedder_name)
            embeddings1.append(emb.embedding);
        for image in images2:
            embeddings2.append(self.emb_manager.get_embedding_by_name(image,detector_name,kinship_embedder_name).embedding);

        kinship_similarity_matrix=cosine_similarity(embeddings1,embeddings2)
        average_similarity = np.mean(kinship_similarity_matrix)

        return average_similarity,len(embeddings1),len(embeddings2)