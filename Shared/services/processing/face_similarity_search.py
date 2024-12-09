from Shared.models.detector_name import DetectorName;
from Shared.models.embedder_name import EmbedderName;
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import numpy as np
from Shared.services.stores.embeddings import BaseImageEmbeddingManager
from Shared.models.errors import FaceEmbeddingError,FaceExtractionError
from Shared.models.errors.base_error import BaseError 
from Shared.services.stores.media import BaseImageStorage
from Shared.models.similar_image import SimilarImage
from .local.local_embedding_generator import LocalEmbeddingGenerator
from .face_aligner import FaceAligner
from Shared.services.models import BaseDetectorModel,BaseEmbedderModel
import time
import os
from Shared.services.util import face_path 
import tempfile

class FaceSimilaritySearch:
    
    def __init__(self,emb_manager:BaseImageEmbeddingManager,
                 embedding_generator:LocalEmbeddingGenerator,
                 image_storage:BaseImageStorage,
                 face_aligner:FaceAligner):
        
        self.emb_manager=emb_manager
        self.embedding_generator=embedding_generator
        self.face_aligner=face_aligner
        self.image_storage=image_storage

    def get_k_similar_images(
        self,
        filename: str,
        selected_face: int,
        similarity_threshold: float,
        detector: BaseDetectorModel,
        embedder: BaseEmbedderModel,
        k=1,
        quality_thresh:float=0,
    ) -> FaceEmbeddingError|FaceExtractionError|tuple[list[SimilarImage]]:
        similar_images = []
        aligned_filename = face_path(filename,selected_face);
        start = time.time()
        embedding = self.emb_manager.get_embedding_by_name(
            aligned_filename, detector_name=detector.name, embedder_name=embedder.name
        )
        if(embedding is None):
            img,file_path=self.image_storage.load_image(filename,detector.name)
            faces_dir=tempfile.TemporaryDirectory(prefix=filename,suffix="_faces")

            det_result = self.face_aligner.create_aligned_images(
                file_path,save_file_name=filename ,img=img,faces_dir=faces_dir.name,detector=detector
            )
            if(isinstance(det_result,BaseError)):
                os.rmdir(faces_dir.name)
                return det_result;
            
            img, faces =det_result
            result = self.embedding_generator.generate_all_emb(
                img, faces, filename, detector, embedder
            )
            if(isinstance(result,BaseError)):
                os.rmdir(faces_dir.name)
                return result;
        
            _, new_embs ,_=result
            self.emb_manager.add_embedding_typed(
                    new_embs, detector.name, embedder.name
                )
            os.rmdir(faces_dir.name)
            embedding = next((x for x in new_embs if x.name==aligned_filename), None)
        end = time.time()
        print(f"Elapsed Get Embedding Time: {(end - start)*1000}ms")
        if embedding and len(embedding.embedding) > 0:
            user_embedding = embedding.embedding
        else:
            user_embedding = self.embedding_generator.generate_embedding(
                filename, selected_face, detector, embedder
            )
            if(isinstance(user_embedding,BaseError)):
                return user_embedding;
        start = time.time()

    
        valid = self.get_similar_images(
            user_embedding,
            filename=filename,
            detector_name=detector.name,
            embedder_name=embedder.name,
            k=k,
            quality_thresh=quality_thresh
        )
        end = time.time()
        print(f"Elapsed Similar Images Time: {(end - start)*1000}ms")
        for image in valid:
            try:
                if image["similarity"]>similarity_threshold:
                    match = image["name"]
                    _, facenum, filename = match.split("_", 2)
                    similar_model = SimilarImage(image_name=filename,face_num=int(facenum),similarity=image["similarity"])
                    similar_images.append(similar_model)
            except Exception as e:
                # template_matching
                return BaseError(reason=f"failed to match image {match} because:\n{e}")

        return similar_images

    def get_similar_images(
        self,
        user_embedding: list,
        filename: str,
        detector_name: DetectorName,
        embedder_name: EmbedderName,
        k=5,
        quality_thresh:float=0
    ):
        np_emb = np.array(user_embedding).astype("float32").reshape(1, -1)

        start = time.time()

        result = self.emb_manager.search(np_emb, k + 1, detector_name, embedder_name,quality_thresh)
        end = time.time()
        print(f"Elapsed Search Time: {(end - start)*1000} ms")
        filtered = []
        seen_distances = []
        for r in result[0]:
            if r["distance"] not in seen_distances:
                seen_distances.append(r["distance"])
                i = r["index"]
                name = r['Embedding'].name
                if name.split("_")[-1] != filename.split("_")[-1]:
                    filtered.append({"index": i,"similarity":r["distance"], "name": name,"Embedding":r['Embedding']})
        valid = [
            x
            for x in filtered
            if len(
                emb := x["Embedding"].embedding
            )
            > 0
            and not np.allclose(emb, np_emb, rtol=1e-5, atol=1e-8)
        ]
        return valid
