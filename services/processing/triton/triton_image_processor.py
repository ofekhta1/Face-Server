from services.models.model_loader import ModelLoader
from services.processing.image_loader import ImageLoader
from models.errors.base_error import BaseError
import numpy as np
import os
from services.processing.local.local_embedding_generator import LocalEmbeddingGenerator
from services.stores import InMemoryImageEmbeddingManager,MilvusImageEmbeddingManager
from services.processing.face_aligner import FaceAligner
import tritonclient.http as httpclient
from models.detector_name import DetectorName
from .triton_client_handler import TritonClientHandler
from insightface.app.common import Face
from models.stored_embedding import FaceEmbedding
from .triton_face_extractor import TritonFaceExtractor
class TritonImageProcessor:

    def __init__(self,emb_manager:InMemoryImageEmbeddingManager|MilvusImageEmbeddingManager,
                 face_aligner:FaceAligner,face_extractor:TritonFaceExtractor,model_loader:ModelLoader):
        self.model_loader=model_loader
        self.emb_manager=emb_manager
        self.face_aligner=face_aligner
        self.face_extractor=face_extractor

    async def process_image(self,file_path:str,save_file_name:str,faces_dir:str)->tuple[dict[str,list[np.ndarray]],list[str]]:
        errors=[]
        generated_embeddings: dict[str, list[np.ndarray]] = {}
        img=ImageLoader.load_image(file_path);
        gender_age_model=self.model_loader.load_genderage("MobileNetCeleb0.25_CelebA");
        input_img_arr=np.array([img])
        input_image = httpclient.InferInput("input_image", input_img_arr.shape, datatype="UINT8")
        input_image.set_data_from_numpy(input_img_arr, binary_data=True)

        response=TritonClientHandler.infer("pipeline",[input_image])
        for detector_name in self.model_loader.model_registry["detectors"]:

            os.makedirs(os.path.join(faces_dir,detector_name),exist_ok=True)


            kpss=response.as_numpy(f"{detector_name}_kps")[0];
            dets=response.as_numpy(f"{detector_name}_dets")[0];
            qualities=response.as_numpy(f"{detector_name}_quality_scores");
            faces=[Face(bbox=det[0:4],kps=kps,det_score=det[4],quality=quality)  for det,kps,quality in zip(dets,kpss,qualities)]
            if len(faces)==0:
                print(f"No faces detected for detector: {detector_name}")
                continue;

            sorted_indices = sorted(range(len(faces)), key=lambda x: faces[x]["quality"], reverse=True)
            faces = [faces[i] for i in sorted_indices]

            genders,ages=gender_age_model.get_gender_age(img,faces)
            aligned_images=[self.face_aligner.align_single_image(face,i,faces_dir,img,detector_name,save_file_name) for i,face in enumerate(faces)]

            for embedder_name in self.model_loader.model_registry["embedders"]:
                embeddings=response.as_numpy(f"{detector_name}_{embedder_name}_embeddings");
                embeddings=[embeddings[i] for i in sorted_indices]#sort by quality,so it matches the faces order
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

                # Normalize each row
                normalized_embeddings = embeddings / norms
                generated_embeddings[f"{detector_name}_{embedder_name}"] = normalized_embeddings

                face_embeddings:list[FaceEmbedding]=[];
                               
                for i,face in enumerate(faces):
                    # face["embedding"]=normalized_embeddings[i] #no need to store embedding cause it is saved as DTO
                    aligned_image_name=aligned_images[i]
                    bbox=[int(coord) for coord in face['bbox']]
                    landmarks=[(x[0],x[1]) for x in face["kps"]]
                    quality=face["quality"] if "quality" in face else 1
                    age=face["age"] if "age" in face else -1
                    gender=face["gender"] if "gender" in face else ""
                    f=FaceEmbedding(aligned_image_name,bbox,normalized_embeddings[i],quality=quality,landmarks=landmarks
                                    ,age=age,gender=gender)
                    face_embeddings.append(f)

                query=np.array([f.embedding for f in face_embeddings])
                similar=self.emb_manager.search(query,1,detector_name,embedder_name)
                distance_dup_thresh=0.95
                for i in range(len(similar)):
                    if len(similar[i])>0:
                        closest=similar[i][0]
                        if closest['distance']> distance_dup_thresh:
                            face_embeddings[i].is_dup=True


                
                self.emb_manager.add_embedding_typed(face_embeddings,detector_name,embedder_name)
        return generated_embeddings,errors

                        