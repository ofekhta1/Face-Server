import sys
import os
import traceback
import insightface
from .face_analysis import FaceAnalysis
from .base_model import BaseModel
sys.path.append(os.path.abspath('..'))
from ..util import are_bboxes_similar


class AntelopeV2(BaseModel):
    def __init__(self,root=""):
        self.name="antelopev2"
        self.embedder= self.__CreateEmbedder(64,root)
        self.detector_zoomed= self.__CreateDetector(320,root)
        self.detector= self.__CreateDetector(1024,root)
    def __CreateEmbedder(self,size,root):
        try:
            base_path=os.path.dirname(sys.argv[0])
            model_name = os.path.join(base_path,"OnnxModels",self.name,"glintr100.onnx")
            embedder = insightface.model_zoo.get_model(model_name)
            embedder.prepare(ctx_id=1, det_thresh=0.5, det_size=(size, size))
            return embedder
        except Exception as e:
            tb = traceback.format_exc()
            print("Error during model initialization:", e)
            return None
    def __CreateDetector(self,size,root):
        try:
            face_analyzer = FaceAnalysis(name="antelopev2",root=root, allowed_modules=["detection","genderage"])
            face_analyzer.prepare(ctx_id=1, det_thresh=0.75, det_size=(size, size))
            return face_analyzer

        except Exception as e:
            tb = traceback.format_exc()
            print("Error during model initialization:", e)
            return None

    def extract_faces(self,img):
        try:
            close_faces=self.detector_zoomed.get(img)
            far_faces=self.detector.get(img)
            faces=far_faces.copy();
            for j in range(len(close_faces)):
                        duplicate=False;
                        for far_face in far_faces:
                            if(are_bboxes_similar(close_faces[j]['bbox'],far_face['bbox'],20)):
                                duplicate=True;
                        if(not duplicate):
                            faces.append(close_faces[j])
            return faces
        except Exception as e:
            return None;
    def embed(self,img,face):
         return self.embedder.get(img,face)