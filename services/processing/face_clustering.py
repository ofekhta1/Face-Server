from models.detector_name import DetectorName;
from models.embedder_name import EmbedderName;
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import numpy as np
from services.stores import ImageGroupRepository,InMemoryImageEmbeddingManager,MilvusImageEmbeddingManager
from services.models.family.family_classifier import FamilyClassifier

class FaceClustering:
    def __init__(self,emb_manager:InMemoryImageEmbeddingManager|MilvusImageEmbeddingManager,
                 groups:ImageGroupRepository):
        self.emb_manager=emb_manager
        self.groups=groups
    def cluster_images(
        self, max_distance:float, min_samples:int, detector_name:DetectorName, embedder_name:EmbedderName,quality_thresh:float=0
    ) -> dict[int, list[str]]:
        # Assuming 'embeddings' is a list of your 512-dimensional embeddings
        face_embeddings =self.emb_manager.get_all_embeddings(detector_name,embedder_name,quality_thresh=quality_thresh)
        embeddings=[e.embedding for e in face_embeddings]
        if len(embeddings) == 0:
            return {}
        #create a similarity matrix that includes the cosine similarity between every 2 images in the embedding data
        similarity_matrix = cosine_similarity(embeddings)
        similarity_matrix = np.clip(similarity_matrix, -1, 1)
        # Apply DBSCAN-model that takes the min sample and max distance as returns the groups according to the required distances and min images in group parameters

        dbscan = DBSCAN(eps=max_distance, min_samples=min_samples, metric="precomputed")
        labels = dbscan.fit_predict(1 - similarity_matrix)  # Convert similarity to distance
        unique_values = np.unique(labels)
        index_groups = {value: np.where(labels == value)[0] for value in unique_values}
        value_groups = {
            str(key): [
                face_embeddings[index].name
                for index in indices
            ]
            for key, indices in index_groups.items()
        }
        self.groups.train_index(value_groups, detector_name, embedder_name)
        self.groups.save_index(detector_name)

        return value_groups;
   
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
            embeddings1.append(self.emb_manager.get_embedding_by_name(image,detector_name,kinship_embedder_name).embedding);
        for image in images2:
            embeddings2.append(self.emb_manager.get_embedding_by_name(image,detector_name,kinship_embedder_name).embedding);

        kinship_similarity_matrix=cosine_similarity(embeddings1,embeddings2)
        average_similarity = np.mean(kinship_similarity_matrix)

        return average_similarity,len(embeddings1),len(embeddings2)