import insightface
import os
from insightface.utils.face_align import transform
from .base_genderage_model import BaseGenderAgeModel
from onnxruntime import InferenceSession
import cv2
import numpy as np
from Shared.services.processing.triton.triton_client_handler import TritonClientHandler
import tritonclient.http as httpclient

class Triton_GenderAge(BaseGenderAgeModel):
    def __init__(self, root=""):
        self.model_name = "GenderAge"
        self.input_size = (96, 96)
        self.input_mean = 0
        self.input_std = 1

    def preprocess(self, img, faces: list[dict] | dict):
        if not isinstance(faces, list):
            faces = [faces]
        aimgs = []
        for face in faces:
            bbox = face.bbox
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = self.input_size[0] / (max(w, h) * 1.5)
            # print('param:', img.shape, bbox, center, self.input_size, _scale, rotate)
            aimg, M = transform(img, center, self.input_size[0], _scale, rotate)
            aimgs.append(aimg)
        # assert input_size==self.input_size
        blob = cv2.dnn.blobFromImages(
            aimgs,
            1.0 / self.input_std,
            self.input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )

        return blob

    def get_gender_age(self, img, faces):
        blob = self.preprocess(img, faces)
        input = httpclient.InferInput("data", blob.shape, datatype="FP32")
        input.set_data_from_numpy(blob, binary_data=True)
        response=TritonClientHandler.infer(model_name=self.model_name,inputs=[input])
        if isinstance(faces, list):
            genders=[]
            ages=[]
            pred=response.as_numpy("fc1")

            for i, result in enumerate(pred):
                gender = "M" if np.argmax(result[:2]) else "W"
                age = int(np.round(result[2] * 100))
                faces[i]["gender"] = gender
                faces[i]["age"] = age
                genders.append(gender)
                ages.append(age)

            return genders, ages

        else:
            pred=response.as_numpy("fc1")[0]
            assert len(pred) == 3
            gender = "M" if np.argmax(pred[:2]) else "W"
            age = int(np.round(pred[2] * 100))
            faces["gender"] = gender
            faces["age"] = age
            return gender, age

