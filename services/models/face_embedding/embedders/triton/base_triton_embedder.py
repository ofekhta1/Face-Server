import os
from models.embedder_name import EmbedderName
from services.models.face_embedding.embedders.base_embedder_model import BaseEmbedderModel
import numpy as np
from insightface.utils.face_align import norm_crop
from services.processing.triton.triton_client_handler import TritonClientHandler
import tritonclient.http as httpclient
from tritonclient.http._infer_result import  InferResult
import cv2 
from insightface.app.common import Face
class BaseTritonEmbedder(BaseEmbedderModel):
    def __init__(self,root=""):
        self.model_name=""
        self.input_name=""
        self.output_name=""
    def embedding_preprocessing(self,img: cv2.Mat,faces:Face|list[Face],input_std=127.5,input_mean=127.5) -> tuple[cv2.Mat,float]:
        
        imgs=[]
        if isinstance(faces,list):
            for face in faces:
                aimg=norm_crop(img,face.kps)
                imgs.append(aimg)
        else:
            #single face
            imgs.append(norm_crop(img,faces.kps))

        input_size =(112,112)
            
        blob = cv2.dnn.blobFromImages(imgs, 1.0 / input_std, input_size,
                                        (input_mean, input_mean, input_mean), swapRB=True)
        return blob
    def embedding_preprocessing_raw(self,img: cv2.Mat|list[cv2.Mat],input_std=127.5,input_mean=127.5) -> tuple[cv2.Mat,float]:
        
        if isinstance(img,list):
            imgs=img
        else:
            imgs=[img]
        input_size =(112,112)
            
        blob = cv2.dnn.blobFromImages(imgs, 1.0 / input_std, input_size,
                                        (input_mean, input_mean, input_mean), swapRB=True)
        return blob
    def embed_raw(self,img:cv2.Mat|list[cv2.Mat]):
        blob=self.embedding_preprocessing_raw(img)
        input = httpclient.InferInput(self.input_name, blob.shape, datatype="FP32")
        input.set_data_from_numpy(blob, binary_data=True)
        response=TritonClientHandler.infer(model_name=self.model_name,inputs=[input])
        embeddings=response.as_numpy(self.output_name);
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Normalize each row
        normalized_embeddings = embeddings / norms
        return normalized_embeddings;

    def embed(self,img,faces)->np.ndarray:
      
        blob=self.embedding_preprocessing(img,faces)
        input = httpclient.InferInput(self.input_name, blob.shape, datatype="FP32")
        input.set_data_from_numpy(blob, binary_data=True)
        response=TritonClientHandler.infer(model_name=self.model_name,inputs=[input])
        embeddings=response.as_numpy(self.output_name);
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Normalize each row
        normalized_embeddings = embeddings / norms
        return normalized_embeddings;
