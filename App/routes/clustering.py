
from fastapi import APIRouter,HTTPException,Depends
from routes.resources import Container
from dependency_injector.wiring import inject, Provide
from Shared.services.processing.face_clustering import FaceClustering
from Shared.services.stores import BaseClusterRepository
from Shared.models.detector_name import DetectorName
from Shared.models.embedder_name import EmbedderName
from App.models.requests import GetClustersRequest, ChangeGroupNameRequest,CompareKinshipClustersRequest,AssignClusterRequest
from App.models.responses import CompareKinshipResponse
from Shared.services.util import face_path
clustering_router=APIRouter()


@clustering_router.post("/api/cluster")
@inject
async def make_clusters(request:GetClustersRequest,
                  face_clustering:FaceClustering=Depends(Provide[Container.face_clustering]))-> dict[str, list[str]]:
    
    eps = request.max_distance
    min_samples = request.min_samples
    quality_thresh=request.quality_threshold
    detector_name = request.detector_name
    embedder_name = request.embedder_name

    value_groups =await face_clustering.cluster_images(
        eps, min_samples, detector_name=detector_name, embedder_name=embedder_name,quality_thresh=quality_thresh,retrain=request.retrain
    )

    return value_groups
   
@clustering_router.post("/api/clustering/assign_group")
@inject
async def assign_group(request:AssignClusterRequest,
               groups:BaseClusterRepository=Depends(Provide[Container.groups])):
    detector_name=request.detector_name
    embedder_name=request.embedder_name
    img_name=face_path(request.image,request.selected_face)
    if(not await groups.has_group(detector_name,embedder_name)):
        raise HTTPException(400,f"Group for Detector:{detector_name} and Embedder:{embedder_name} does not exist!") 
    
    new_group=await groups.assign(img_name,request.cluster_id,detector_name,embedder_name);
    return {"new_group":new_group}

@clustering_router.get("/api/clustering/get_groups")
@inject
async def get_groups(detector_name:DetectorName,embedder_name:EmbedderName,
               groups:BaseClusterRepository=Depends(Provide[Container.groups]))-> dict[str, list[str]]:

    if(not await groups.has_group(detector_name,embedder_name)):
        raise HTTPException(400,f"Group for Detector:{detector_name} and Embedder:{embedder_name} does not exist!") 
    
    id_groups=await groups.get_id_groups(detector_name,embedder_name);
    return id_groups

@clustering_router.get("/api/clustering/get_thumbnails")
@inject
async def get_groups_thumbnails(detector_name:DetectorName,embedder_name:EmbedderName,
                           groups:BaseClusterRepository=Depends(Provide[Container.groups]))-> dict[str,str]:
    thumbnails: dict[str, str] = await groups.get_thumbnails(detector_name,embedder_name)
    if thumbnails is None:
        return {}
    return thumbnails

@clustering_router.post("/api/clustering/change_group_name")
@inject
async def change_group_name(request:ChangeGroupNameRequest,
                       groups:BaseClusterRepository=Depends(Provide[Container.groups])):

    detector_name = request.detector_name
    embedder_name = request.embedder_name
    new_group=await groups.change_group_name(request.old, request.new, detector_name, embedder_name)
    return {"new_group":new_group}


@clustering_router.get("/api/clustering/get_group_images")
@inject
async def get_group_images(detector_name:DetectorName,embedder_name:EmbedderName,group_id:str,
                     groups:BaseClusterRepository=Depends(Provide[Container.groups])):
    images=await groups.get_by_id(detector_name,embedder_name,group_id)
    return {"images":images}


@clustering_router.post("/api/clustering/compare_kinship_clusters")
@inject
async def compare_kinship_clusters(request:CompareKinshipClustersRequest,
                     face_clustering:FaceClustering=Depends(Provide[Container.face_clustering])):
    result=await face_clustering.compare_kinship_clusters(request.cluster_id_1,request.cluster_id_2,request.detector_name,request.embedder_name,
                                                    request.kinship_embedder_name);
    average_similarities={
        request.kinship_embedder_name: result[0]
    }
    response =CompareKinshipResponse(average_similarities=average_similarities,cluster1_count=result[1],cluster2_count=result[2])
    return response;