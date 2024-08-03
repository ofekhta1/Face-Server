from services.models.face_embedding.detectors.base_detector_model import BaseDetectorModel
from models.errors import FaceExtractionError,FaceEmbeddingError
from services.processing.local.local_face_extractor import LocalFaceExtractor
from models.stored_embedding import FaceEmbedding
from services.models.face_embedding.embedders.base_embedder_model import BaseEmbedderModel
from services.models.face_embedding.genderage.base_genderage_model import BaseGenderAgeModel
import numpy as np
import os
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from config.app_paths import AppPaths
from services.stores.in_memory_image_embedding_manager import InMemoryImageEmbeddingManager

class LocalEmbeddingGenerator:

    def __init__(self,emb_manager:InMemoryImageEmbeddingManager,face_extractor:LocalFaceExtractor) :
        self.face_extractor=face_extractor
        self.emb_manager=emb_manager
    
    
    def generate_all_emb(
        self,
        filename: str,
        detector: BaseDetectorModel,
        embedder: BaseEmbedderModel,
        save=True,
    ) -> FaceEmbeddingError|tuple[np.typing.NDArray[np.uint8], list[FaceEmbedding] ]:
        errors = []
        embeddings = []
        if detector:
            img, faces = self.face_extractor.extract_faces(filename, detector)
            return self.generate_all_emb(img, faces, filename, detector, embedder, save)
        errors.append("Error: detector model not initialized.")
        return embeddings, errors
   
    def generate_all_emb(
        self,
        img,
        faces: list,
        filename: str,
        detector: BaseDetectorModel,
        embedder: BaseEmbedderModel,
        gender_age:BaseGenderAgeModel=None
    ) -> FaceEmbeddingError|tuple[np.typing.NDArray[np.uint8], list[FaceEmbedding]]:
        if not faces:
            return FaceEmbeddingError(reason="Faces not exctracted",embedder_name=embedder.name)
        faces=faces.copy();
        aligned_images = []
        embeddings = []

        for i in range(len(faces)):
            try:
                embedding = embedder.embed(img, faces[i])
                norm = np.linalg.norm(embedding)
                embedding=embedding/norm;
                embeddings.append(embedding)
                aligned_filename = f"aligned_{i}_{filename}"

                aligned_images.append(aligned_filename)
            except Exception as ex:
                return FaceEmbeddingError(embedder_name=embedder.name)

        # internal dedup code

        similarity_matrix = cosine_similarity(embeddings)
        similarity_matrix = np.clip(similarity_matrix, -1, 1)
        # Apply DBSCAN

        dbscan = DBSCAN(eps=0.1, min_samples=2, metric="precomputed")
        labels = dbscan.fit_predict(
            1 - similarity_matrix
        )  # Convert similarity to distance
        clusters = np.unique(labels)
        clusters = np.delete(
            clusters, np.where(clusters == -1)
        )  # remove -1, non cluster data
        index_groups = {value: np.where(labels == value)[0] for value in clusters}
        for group in index_groups:
            # remove all values except for first value per cluster(dups)
            for dups_index in index_groups[group][1:]:
                dup_path=os.path.join(AppPaths.STATIC_FOLDER, detector.name, aligned_images[dups_index])
                if os.path.exists(dup_path):
                    os.remove(dup_path)
                embeddings[dups_index] = None
                faces[dups_index] = None
                aligned_images[dups_index] = None

        filtered_embeddings = [e for e in embeddings if e is not None]
        filtered_faces = [f for f in faces if f is not None]
        filtered_aligned_images = [ai for ai in aligned_images if ai is not None]
        face_embeddings:list[FaceEmbedding]=[]
        for i in range(len(filtered_faces)):
            bbox=[int(coord) for coord in filtered_faces[i]['bbox']]
            landmarks=[(x[0],x[1]) for x in filtered_faces[i]["kps"]]
            quality=filtered_faces[i]["quality"] if "quality" in filtered_faces[i] else 1
            f=FaceEmbedding(filtered_aligned_images[i],bbox,filtered_embeddings[i],quality=quality,landmarks=landmarks)
            if(gender_age):
                f.gender,f.age=gender_age.get_gender_age(img,filtered_faces[i]);          
            face_embeddings.append(f);
        #check if embeddings exist in emb_manager,in batch
        query=np.array([f.embedding for f in face_embeddings])
        similar=self.emb_manager.search(query,1,detector.name,embedder.name)
        distance_dup_thresh=0.95
        for i in range(len(similar)):
            closest=similar[i][0]
            if closest['distance']> distance_dup_thresh:
                face_embeddings[i].is_dup=True

        
        # self.emb_manager.set_face_count(filename,len(filtered_faces),detector_name=model.name)
        return img, face_embeddings,filtered_faces

    def generate_embedding(
        self,
        filename: str,
        selected_face: int,
        detector: BaseDetectorModel,
        embedder: BaseEmbedderModel,
        save=True,
    ) -> np.ndarray[np.float32]|FaceExtractionError| FaceEmbeddingError:
        embedding = None
        img, faces = self.face_extractor.extract_faces(filename, detector)
        if not faces:
            return FaceExtractionError(detector.name,reason="Failed to extract faces");
        if selected_face == -2 or len(faces) == 1:
            i = 0
        else:
            i = selected_face

        embedding = embedder.embed(img, faces[i])
        if embedding is None or len(embedding)==0:
            return FaceEmbeddingError(embedder.name)
        if save:
            box = faces[i]["bbox"].astype(int).tolist()
            self.emb_manager.add_embedding(
                embedding,
                f"aligned_{i}_{filename}",
                box,
                detector.name,
                embedder.name,
            )
            
        return embedding
