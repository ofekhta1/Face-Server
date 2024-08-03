import sys
import os
import traceback
from .retinaface50.retinaface import RetinaFace 
from .base_detector_model import BaseDetectorModel
sys.path.append(os.path.abspath('..'))
from services.util import are_bboxes_similar
sys.path.append(os.path.abspath('../..'))
from insightface.app.common import Face
import cv2
import numpy as np
from models.detector_name import DetectorName

class EranRetinaFaceDetector(BaseDetectorModel):
    def __init__(self,root=""):
        self.name=DetectorName.eran_retinaface
        self.model_name = "" # Use the face recognition model
        self.detector= self.CreateDetector(root)
        self.input_size_zoomed=(320,320)
        self.input_size=(1024,1024)
        self.face_ratio_thresh=0.001
    def CreateDetector(self,root):
        try:
            detector = RetinaFace(os.path.join(root,"services/models/face_embedding/detectors/retinaface50/R50"), 0)
            return detector;
        except Exception as e:
            tb = traceback.format_exc()
            print("Error during model initialization:", e)
            return None
    def __extract_faces_internal(self,img,input_size,resize=False):
        try:
            if(resize):
                im_ratio = float(img.shape[0]) / img.shape[1]
                model_ratio = float(input_size[1]) / input_size[0]
                if im_ratio>model_ratio:
                    new_height = input_size[1]
                    new_width = int(new_height / im_ratio)
                else:
                    new_width = input_size[0]
                    new_height = int(new_width * im_ratio)
                    
                det_scale = float(new_height) / img.shape[0]
                resized_img = cv2.resize(img, (new_width, new_height))
                det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
                det_img[:new_height, :new_width, :] = resized_img
            else:
                det_scale=1
                det_img=img;
            bboxes, landmarks = self.detector.detect(det_img,
                                    0.8,
                                    scales=[1.0],
                                    do_flip=False)

            bboxes_scaled = bboxes / det_scale
            landmarks_scaled = landmarks / det_scale
            faces=[Face(bbox=bbox[0:4],kps=kps,det_score=bbox[4]) for bbox,kps in zip(bboxes_scaled,landmarks_scaled)]


            return faces
        except Exception as e:
            print("Error during face extraction:", e)
            return None;

    def extract_faces(self,img:np.ndarray):
        faces=self.__extract_faces_internal(img,self.input_size)
        # far_faces=self.__extract_faces_internal(img,self.input_size_zoomed)
        # faces=far_faces.copy(); 
        # for j in range(len(close_faces)):
        #     duplicate=False;
        #     for far_face in far_faces:
        #         if(are_bboxes_similar(close_faces[j]['bbox'],far_face['bbox'],20)):
        #             duplicate=True;
        #     if(not duplicate):
        #         faces.append(close_faces[j])
 
        return faces