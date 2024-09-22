
from fastapi import APIRouter,HTTPException,Depends
from routes.resources import Container
from dependency_injector.wiring import inject, Provide
from services.processing.face_clustering import FaceClustering
from services.stores import ImageGroupRepository
from models.detector_name import DetectorName
from models.embedder_name import EmbedderName
from models.requests import GetClustersRequest, ChangeGroupNameRequest,CompareKinshipClustersRequest,AssignClusterRequest
from models.responses import CompareKinshipResponse

clustering_router=APIRouter()


@clustering_router.post("/api/cluster")
@inject
def make_clusters(request:GetClustersRequest,
                  face_clustering:FaceClustering=Depends(Provide[Container.face_clustering]))-> dict[str, list[str]]:
    
    eps = request.max_distance
    min_samples = request.min_samples
    quality_thresh=request.quality_threshold
    detector_name = request.detector_name
    embedder_name = request.embedder_name

    value_groups = face_clustering.cluster_images(
        eps, min_samples, detector_name=detector_name, embedder_name=embedder_name,quality_thresh=quality_thresh,retrain=request.retrain
    )

    return value_groups
   
@clustering_router.post("/api/assign_group")
@inject
def assign_group(request:AssignClusterRequest,
               groups:ImageGroupRepository=Depends(Provide[Container.groups])):
    detector_name=request.detector_name
    embedder_name=request.embedder_name
    img_name=f"aligned_{request.selected_face}_{request.image}"
    if(not groups.has_group(detector_name,embedder_name)):
        raise HTTPException(400,f"Group for Detector:{detector_name} and Embedder:{embedder_name} does not exist!") 
    
    new_group=groups.assign(img_name,request.cluster_id,detector_name,embedder_name);
    return {"new_group":new_group}

@clustering_router.get("/api/get_groups")
@inject
def get_groups(detector_name:DetectorName,embedder_name:EmbedderName,
               groups:ImageGroupRepository=Depends(Provide[Container.groups]))-> dict[str, list[str]]:

    if(not groups.has_group(detector_name,embedder_name)):
        raise HTTPException(400,f"Group for Detector:{detector_name} and Embedder:{embedder_name} does not exist!") 
    
    id_groups=groups.get_id_groups(detector_name,embedder_name);
    return id_groups

@clustering_router.get("/api/get_thumbnails")
@inject
def get_groups_thumbnails(detector_name:DetectorName,embedder_name:EmbedderName,
                           groups:ImageGroupRepository=Depends(Provide[Container.groups]))-> dict[str,str]:
    thumbnails: dict[str, str] = groups.get_thumbnails(detector_name,embedder_name)
    return thumbnails

@clustering_router.post("/api/change_group_name")
@inject
def change_group_name(request:ChangeGroupNameRequest,
                       groups:ImageGroupRepository=Depends(Provide[Container.groups])):

    detector_name = request.detector_name
    embedder_name = request.embedder_name
    new_group=groups.change_group_name(request.old, request.new, detector_name, embedder_name)
    return {"new_group":new_group}


@clustering_router.get("/api/get_group_images")
@inject
def get_group_images(detector_name:DetectorName,embedder_name:EmbedderName,group_id:str,
                     groups:ImageGroupRepository=Depends(Provide[Container.groups])):
    images=groups.get_by_id(detector_name,embedder_name,group_id)
    return {"images":images}


@clustering_router.post("/api/compare_kinship_clusters")
@inject
def compare_kinship_clusters(request:CompareKinshipClustersRequest,
                     face_clustering:FaceClustering=Depends(Provide[Container.face_clustering])):
    result=face_clustering.compare_kinship_clusters(request.cluster_id_1,request.cluster_id_2,request.detector_name,request.embedder_name,
                                                    request.kinship_embedder_name);
    
    response =CompareKinshipResponse(average_similarity=result[0],cluster1_count=result[1],cluster2_count=result[2])
    return response;