from Shared.services import util
from App.models.requests import GetImageMetadataRequest,UpdateMetadataRequest
from App.models.responses import GetFacesInfoResponse,GetDetectorIndicesResponse 
from Shared.services.stores import MetadataManager,BaseClusterRepository
from Shared.services.stores.media import BaseImageStorage
from dependency_injector.wiring import inject, Provide
from routes.resources import Container
from fastapi import APIRouter,HTTPException,Depends
image_metadata_router=APIRouter()

@image_metadata_router.post("/api/metadata/get_face_info")
@inject
async def get_face_info(request:GetImageMetadataRequest,
    groups:BaseClusterRepository=Depends(Provide[Container.groups]),
    image_storage:BaseImageStorage=Depends(Provide[Container.image_storage]),
    metadata_manager:MetadataManager=Depends(Provide[Container.metadata_manager]))->GetFacesInfoResponse:
    face_num=request.selected_face
    filename = request.image
    detector_name = request.detector_name
    embedder_name = request.embedder_name
    faces_length = 0
    errors = []
    faces=[]
    

    if face_num==-2:
        path =  filename
    else:
        path = util.face_path(filename,face_num)

    

    if image_storage.file_exists(path,detector_name,False):
        faces = metadata_manager.get_image_faces(
            filename, face_num,detector_name=detector_name, embedder_name=embedder_name
        )
        faces_length = len(faces)
        
        if(request.get_group_id):
            if face_num!=-2:
                   faces[0].group_id=await groups.get_group_id(util.face_path(filename,face_num),detector_name,embedder_name)
            else:
                for i,face in enumerate(faces):
                    face.group_id=await  groups.get_group_id(util.face_path(filename,i),detector_name,embedder_name)
    else:
        errors.append(f"File {filename} does not exist!")
    
    
    return GetFacesInfoResponse(faces=faces,faces_length=faces_length,errors=errors)


@image_metadata_router.post("/api/metadata/get_indices")
@inject
def get_detector_indices(request:GetImageMetadataRequest,
    metadata_manager:MetadataManager=Depends(Provide[Container.metadata_manager])):
    filename = request.image
    return_detector= request.detector_name
   
    detector_indices,detector_metadata =metadata_manager.get_detector_indices(filename,return_detector)
    return GetDetectorIndicesResponse(detector_indices=detector_indices,detector_metadata=detector_metadata)


@image_metadata_router.post("/api/metadata/update_metadata")
@inject
async def update_metadata(request:UpdateMetadataRequest,
    metadata_manager:MetadataManager=Depends(Provide[Container.metadata_manager])):
    if request.selected_face<0:
        raise HTTPException(status_code=400, detail="Selected face must be a positive integer!")
    
    metadata_manager.update_metadata(request.image,request.selected_face,request.metadata,request.detector_name)
    return {"success": True}