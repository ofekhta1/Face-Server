import sys
import os
import traceback
import insightface
from ..base_detector_model import BaseDetectorModel
from Shared.services.util import filter_faces
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
from Shared.models.face import Face


class BaseInsightfaceDetector(BaseDetectorModel):
    def __init__(self,root=""):
        self.name="base_insightface_det"
        self.model_name = "" # Use the face recognition model
        self.detector_zoomed= self.CreateDetector(320,root)
        self.detector= self.CreateDetector(1024,root)
   
    def CreateDetector(self,size,root):
        try:
            detector = insightface.model_zoo.get_model(self.model_name)
            detector.prepare(ctx_id=1, det_thresh=0.75, input_size=(size, size))
            return detector;
        except Exception as e:
            tb = traceback.format_exc()
            print("Error during model initialization:", e)
            return None
  
    def extract_faces(self,img):
        try:
            close_results=self.detector_zoomed.detect(img,
                                             max_num=100,
                                             metric='default')
            far_results=self.detector.detect(img,
                                        max_num=100,
                                        metric='default')
          
            close_faces=[Face(bbox=bbox[0:4],kps=kps,det_score=bbox[4]) for bbox,kps in zip(close_results[0],close_results[1])]
            far_faces=[Face(bbox=bbox[0:4],kps=kps,det_score=bbox[4]) for bbox,kps in zip(far_results[0],far_results[1])]

            return filter_faces(close_faces,far_faces)
        except Exception as e:
            print("Error during face extraction:", e)
            return None;
   
   
   
    