from insightface.app import FaceAnalysis
import traceback 
import insightface

class ModelLoader:

    @staticmethod
    def load_detector(size=640):
        try:
            face_analyzer = FaceAnalysis(name="buffalo_l", allowed_modules=["detection","genderage"])
            face_analyzer.prepare(ctx_id=1, det_thresh=0.75, det_size=(size, size))
            return face_analyzer

        except Exception as e:
            tb = traceback.format_exc()
            print("Error during model initialization:", e)
            return None

    @staticmethod
    def load_embedder(size=640):
        try:
            # model_name = "arcface_r100_v1"  # Use the face recognition model
            # embedder = FaceAnalysis(model=model_name)
            model_name = 'Models/w600k_r50.onnx' # Use the face recognition model
            embedder = insightface.model_zoo.get_model(model_name)
            embedder.prepare(ctx_id=1, det_thresh=0.5, det_size=(size, size))
            return embedder

        except Exception as e:
            print("Error during embedder model initialization:", e)
            return None
    

