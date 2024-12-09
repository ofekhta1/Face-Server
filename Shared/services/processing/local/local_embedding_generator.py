from Shared.services.models.face_embedding.detectors.base_detector_model import BaseDetectorModel
from Shared.models.errors import FaceExtractionError,FaceEmbeddingError
from .local_face_extractor import LocalFaceExtractor
from Shared.services.stores.media.base_image_storage import BaseImageStorage
from Shared.models.stored_embedding import FaceEmbedding
from Shared.services.models.face_embedding.embedders.base_embedder_model import BaseEmbedderModel
import numpy as np
import os
from Shared.services.util import face_path
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from Shared.services.stores.embeddings import BaseImageEmbeddingManager


class LocalEmbeddingGenerator:

    def __init__(self,emb_manager:BaseImageEmbeddingManager,face_extractor:LocalFaceExtractor,image_storage:BaseImageStorage) :
        self.face_extractor=face_extractor
        self.image_storage=image_storage
        self.emb_manager=emb_manager
    

    def generate_all_emb_for_file(
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
        faces_dir: str,
        detector: BaseDetectorModel,
        embedder: BaseEmbedderModel,
        filename=""
    ) -> FaceEmbeddingError|tuple[np.typing.NDArray[np.uint8], list[FaceEmbedding]]:
        if not faces:
            return FaceEmbeddingError(reason="Faces not exctracted",embedder_name=embedder.name)
        faces=faces.copy();
        aligned_images = []

        try:
            embeddings = embedder.embed(img, faces)
            aligned_images=[face_path(filename,i) for i in range(len(embeddings))]

        except Exception as ex:
            return FaceEmbeddingError(embedder_name=embedder.name,reason=str(ex))

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
                self.image_storage.remove_media_file(aligned_images[dups_index],detector.name)
                embeddings[dups_index] = None
                faces[dups_index] = None
                aligned_images[dups_index] = None

        filtered_data = [(e, f, ai) for e, f, ai in zip(embeddings, faces, aligned_images) if not np.isnan(e[0])  and f is not None and ai is not None]
        filtered_embeddings, filtered_faces, filtered_aligned_images = map(list, zip(*filtered_data)) if filtered_data else ([], [], [])
        
        # if there are duplicates then reindex the faces
        if len(index_groups)!=0:
            for idx,fimg in enumerate(filtered_aligned_images):
                original_path=os.path.join(faces_dir, detector.name,fimg)
                parts=fimg.split('_',2);
                fimg_filename=parts[-1];
                new_name=face_path(fimg_filename,idx)
                new_path=os.path.join(faces_dir, detector.name,new_name )
                os.rename(original_path,new_path)
                filtered_aligned_images[idx]=new_name
        face_embeddings:list[FaceEmbedding]=[]

        for i,face in enumerate(filtered_faces):
            bbox=[int(coord) for coord in face['bbox']]
            landmarks=[(x[0],x[1]) for x in face["kps"]]
            quality=face["quality"] if "quality" in face else 1
            f=FaceEmbedding(filtered_aligned_images[i],bbox,filtered_embeddings[i],gender=face.gender,age=face.age,quality=quality,landmarks=landmarks)
            face_embeddings.append(f);
        
        #check if embeddings exist in emb_manager,in batch
        query=np.array([f.embedding for f in face_embeddings])
        similar=self.emb_manager.search(query,1,detector.name,embedder.name)
        distance_dup_thresh=0.95
        for i in range(len(similar)):
            if len(similar[i])>0:
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
            return FaceExtractionError(detector_name=detector.name,reason="Failed to extract faces");
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
                face_path(filename, i),
                box,
                detector.name,
                embedder.name,
            )
            
        return embedding
