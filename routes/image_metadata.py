from services import util
from services.models.model_loader import ModelLoader
from config.app_paths import AppPaths
import os
from models.requests import GetImageMetadataRequest
from models.responses import GetFacesInfoResponse,GetDetectorIndicesResponse 
from services.stores import MetadataManager,ImageGroupRepository
from dependency_injector.wiring import inject, Provide
from routes.resources import Container
from fastapi import APIRouter,HTTPException,Depends
import numpy as np
image_metadata_router=APIRouter()

@image_metadata_router.post("/api/get_face_info")
@inject
def get_face_info(request:GetImageMetadataRequest,
    groups:ImageGroupRepository=Depends(Provide[Container.groups]),
    metadata_manager:MetadataManager=Depends(Provide[Container.metadata_manager]))->GetFacesInfoResponse:
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
        path = os.path.join(AppPaths.STATIC_FOLDER,request.detector_name.value, util.face_path(filename,face_num))

    if os.path.exists(path):
        faces = metadata_manager.get_image_faces(
            filename, face_num,detector_name=detector_name, embedder_name=embedder_name
        )
        faces_length = len(faces)
        
        if(request.get_group_id):
            if face_num!=-2:
               faces[0].group_id=groups.get_group_id(util.face_path(filename,face_num),detector_name,embedder_name)
            else:
                for i,face in enumerate(faces):
                    face.group_id=groups.get_group_id(util.face_path(filename,i),detector_name,embedder_name)
    else:
        errors.append(f"File {filename} does not exist!")
    
    
    return GetFacesInfoResponse(faces=faces,faces_length=faces_length,errors=errors)


@image_metadata_router.post("/api/get_indices")
@inject
def get_detector_indices(request:GetImageMetadataRequest,
    metadata_manager:MetadataManager=Depends(Provide[Container.metadata_manager])):
    filename = request.image
    return_detector= request.detector_name
   
    detector_indices =metadata_manager.get_detector_indices(filename,return_detector)
    return GetDetectorIndicesResponse(detector_indices=detector_indices)