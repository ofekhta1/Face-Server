from .base_insightface_detector import BaseInsightfaceDetector
import os

class SCRFD10G(BaseInsightfaceDetector):
    def __init__(self,root=""):
        self.name="SCRFD10G"
        self.model_name = os.path.join(root,"OnnxModels","Detectors","scrfd_10g_bnkps.onnx") # Use the face recognition model
        self.detector_zoomed= self.CreateDetector(320,root)
        self.detector= self.CreateDetector(1024,root)