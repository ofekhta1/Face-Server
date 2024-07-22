from modules import AppPaths,util,ModelLoader
from . import resources
import os
from models.requests import GetImageMetadataRequest
from models.responses import FindFaceResponse,GetDetectorIndicesResponse 
from fastapi import APIRouter,HTTPException
import numpy as np

image_metadata_router=APIRouter()

@image_metadata_router.post("/api/find")
def find_face_in_image(request:GetImageMetadataRequest):
    helper=resources.helper
    
    filename = request.image
    detector_name = request.detector_name
    embedder_name = request.embedder_name

    faces_length = [0]
    messages = []
    errors = []
    boxes = []
    if "aligned" in filename or "detected" in filename:
        path = os.path.join(AppPaths.STATIC_FOLDER, filename)
    else:
        path = os.path.join(AppPaths.UPLOAD_FOLDER, filename)
    if os.path.exists(path):
        boxes = helper.get_face_boxes(
            filename, detector_name=detector_name, embedder_name=embedder_name
        )
        faces_length = len(boxes)
        messages.append(f"{faces_length} detected faces in {filename}.")
    else:
        errors.append(f"File {filename} does not exist!")
    
    return FindFaceResponse(boxes=boxes,faces_length=faces_length,errors=errors)



@image_metadata_router.post("/api/get_indices")
def get_detector_indices(request:GetImageMetadataRequest):
    helper=resources.helper
    manager=resources.manager
    filename = request.image
    generated_embeddings = {}
    return_detector= request.detector_name
    embedder_name=next(iter(ModelLoader.embedders))
    for detector_name in ModelLoader.detectors:
        embs = helper.emb_manager.get_image_embeddings(
                filename, detector_name, embedder_name
            )
        if len(embs) == 0:
            temp_detector=ModelLoader.load_detector(detector_name)
            temp_embedder=ModelLoader.load_embedder(embedder_name)
            img, faces = helper.create_aligned_images(
                filename, temp_detector, []
            )
            if img is not None and faces is not None:
                _, new_embs, _ = helper.generate_all_emb(
                    img, faces, filename, temp_detector, temp_embedder
                )
                manager.add_embedding_typed(
                        new_embs, detector_name, embedder_name
                    )
                embs=np.array([f.embedding for f in new_embs])

        generated_embeddings[f"{detector_name}_{embedder_name}"] = embs

    detector_indices = util.get_all_detectors_faces(
        generated_embeddings, return_detector
    )    
    return GetDetectorIndicesResponse(detector_indices=detector_indices)