from .base_insightface_detector import BaseInsightfaceDetector
import os

class RetinaFace10GF(BaseInsightfaceDetector):
    def __init__(self,root=""):
        self.name="RetinaFace10GF"
        self.model_name = os.path.join(root,"OnnxModels","Detectors","det_10g.onnx") # Use the face recognition model
        self.detector_zoomed= self.CreateDetector(320,root)
        self.detector= self.CreateDetector(1024,root)
   