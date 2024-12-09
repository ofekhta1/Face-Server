import insightface
import os
from insightface.utils.face_align import transform
from .base_genderage_model import BaseGenderAgeModel
from onnxruntime import InferenceSession
import cv2
import numpy as np


class MobileNet_CelebA(BaseGenderAgeModel):
    def __init__(self, root=""):
        self.name = "MobileNetCeleb0.25_CelebA"
        self.model_name = os.path.join(
            root, "OnnxModels", "GenderAge", "genderage.onnx"
        )  # Use the face recognition model
        self.session = InferenceSession(
            self.model_name, providers=BaseGenderAgeModel.providers
        )
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
        input_size = tuple(aimg.shape[0:2][::-1])
        # assert input_size==self.input_size
        blob = cv2.dnn.blobFromImages(
            aimgs,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )

        return blob

    def get_gender_age(self, img, faces):
        blob = self.preprocess(img, faces)
        if isinstance(faces, list):
            genders = []
            ages = []
            pred = self.session.run(["fc1"], {"data": blob})[0]
            for i, result in enumerate(pred):
                gender = "M" if np.argmax(result[:2]) else "W"
                age = int(np.round(result[2] * 100))
                faces[i]["gender"] = gender
                faces[i]["age"] = age
                genders.append(gender)
                ages.append(age)

            return genders, ages

        else:
            pred = self.session.run(["fc1"], {"data": blob})[0][0]
            assert len(pred) == 3
            gender = "M" if np.argmax(pred[:2]) else "W"
            age = int(np.round(pred[2] * 100))
            faces["gender"] = gender
            faces["age"] = age
            return gender, age

    def get_gender_age_raw(self, img)->tuple[str,int]:

        if isinstance(img, list):
            blob = cv2.dnn.blobFromImages(
                img,
                1.0 / self.input_std,
                self.input_size,
                (self.input_mean, self.input_mean, self.input_mean),
                swapRB=True,
            )
            genders = []
            ages = []
            pred = self.session.run(["fc1"], {"data": blob})[0]
            for i, result in enumerate(pred):
                gender = "M" if np.argmax(result[:2]) else "W"
                age = int(np.round(result[2] * 100))
                genders.append(gender)
                ages.append(age)

            return genders, ages

        else:
            blob = cv2.dnn.blobFromImage(
                img,
                1.0 / self.input_std,
                self.input_size,
                (self.input_mean, self.input_mean, self.input_mean),
                swapRB=True,
            )
            pred = self.session.run(["fc1"], {"data": blob})[0][0]
            assert len(pred) == 3
            gender = "M" if np.argmax(pred[:2]) else "W"
            age = int(np.round(pred[2] * 100))
            return gender, age
