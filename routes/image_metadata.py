from services import util,ModelLoader
from config.app_paths import AppPaths
import os
from models.requests import GetImageMetadataRequest
from models.responses import FindFaceResponse,GetDetectorIndicesResponse 
from services.stores import MetadataManager
from dependency_injector.wiring import inject, Provide
from routes.resources import Container
from fastapi import APIRouter,HTTPException,Depends
import numpy as np

image_metadata_router=APIRouter()

@image_metadata_router.post("/api/find")
@inject
def find_face_in_image(request:GetImageMetadataRequest,
    metadata_manager:MetadataManager=Depends(Provide[Container.metadata_manager])):

    face_num=request.selected_face
    filename = request.image
    detector_name = request.detector_name
    embedder_name = request.embedder_name

    faces_length = 0
    errors = []
    faces=[]
    if face_num==-2:
        path = os.path.join(AppPaths.UPLOAD_FOLDER, filename)
    else:
        path = os.path.join(AppPaths.STATIC_FOLDER,request.detector_name.value, f"aligned_{face_num}_{filename}")

    if os.path.exists(path):
        faces = metadata_manager.get_image_faces(
            filename, face_num,detector_name=detector_name, embedder_name=embedder_name
        )
        faces_length = len(faces)
    else:
        errors.append(f"File {filename} does not exist!")
    
    return FindFaceResponse(faces=faces,faces_length=faces_length,errors=errors)



@image_metadata_router.post("/api/get_indices")
@inject
def get_detector_indices(request:GetImageMetadataRequest,
    metadata_manager:MetadataManager=Depends(Provide[Container.metadata_manager])):
    filename = request.image
    return_detector= request.detector_name
   
    detector_indices =metadata_manager.get_detector_indices(filename,return_detector)
    return GetDetectorIndicesResponse(detector_indices=detector_indices)