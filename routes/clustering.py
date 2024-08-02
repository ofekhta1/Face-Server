
from fastapi import APIRouter,HTTPException
from modules import AppPaths
from . import resources
from models.detector_name import DetectorName
from models.embedder_name import EmbedderName
from models.requests import GetClustersRequest, ChangeGroupNameRequest,BaseRequest,CompareKinshipClustersRequest
from models.responses import CompareKinshipResponse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from modules import ModelLoader
clustering_router=APIRouter()


@clustering_router.post("/api/cluster")
def make_clusters(request:GetClustersRequest)-> dict[str, list[str]]:
    helper=resources.helper
    groups=resources.groups
    
    eps = request.max_distance
    min_samples = request.min_samples
    retrain = request.retrain
    quality_thresh=request.quality_threshold
    detector_name = request.detector_name
    embedder_name = request.embedder_name

    value_groups = helper.cluster_images(
        eps, min_samples, detector_name=detector_name, embedder_name=embedder_name,quality_thresh=quality_thresh
    )
    if retrain:
        groups.train_index(value_groups, detector_name, embedder_name)
        groups.save_index(detector_name)
        return value_groups
    
    modified_group: dict[str, list[str]] = {}
    if detector_name  in groups.groups and embedder_name  in groups.groups[detector_name].groups:
        modified_group: dict[str, list[str]] = groups.get_id_groups(detector_name,embedder_name)
     
    return modified_group

@clustering_router.get("/api/get_groups")
def get_groups(detector_name:DetectorName,embedder_name:EmbedderName)-> dict[str, list[str]]:
    groups=resources.groups

    if(not groups.has_group(detector_name,embedder_name)):
        raise HTTPException(400,f"Group for Detector:{detector_name} and Embedder:{embedder_name} does not exist!") 
    
    id_groups=groups.get_id_groups(detector_name,embedder_name);
    return id_groups

@clustering_router.get("/api/get_thumbnails")
def get_groups_thumbnails(detector_name:DetectorName,embedder_name:EmbedderName)-> dict[str,str]:
    groups=resources.groups
    thumbnails: dict[str, str] = groups.get_thumbnails(detector_name,embedder_name)
    return thumbnails
@clustering_router.post("/api/change_group_name")
def change_group_name(request:ChangeGroupNameRequest):
    groups=resources.groups

    detector_name = request.detector_name
    embedder_name = request.embedder_name
    groups.change_group_name(request.old, request.new, detector_name, embedder_name)
    groups.save_index(detector_name)
    return {"success":True}


@clustering_router.get("/api/get_group_images")
def get_group_images(detector_name:DetectorName,embedder_name:EmbedderName,group_id:str):
    groups=resources.groups
    images=groups.get_by_id(detector_name,embedder_name,group_id)
    return {"images":images}

@clustering_router.post("/api/compare_kinship_clusters")
def compare_kinship_clusters(request:CompareKinshipClustersRequest):
    helper=resources.helper
    groups=resources.groups
    
    images1=groups.get_by_id(request.detector_name,request.embedder_name,request.cluster_id_1);
    images2=groups.get_by_id(request.detector_name,request.embedder_name,request.cluster_id_2);
    embeddings1=[]; 
    embeddings2=[];
    for image in images1:
        embeddings1.append(helper.emb_manager.get_embedding_by_name(image,request.detector_name,request.embedder_name).embedding);
    for image in images2:
        embeddings2.append(helper.emb_manager.get_embedding_by_name(image,request.detector_name,request.embedder_name).embedding);

    kinship_similarity_matrix=cosine_similarity(embeddings1,embeddings2)
    average_similarity = np.mean(kinship_similarity_matrix)
    print(kinship_similarity_matrix)

    response =CompareKinshipResponse(average_similarity=average_similarity,cluster1_count=len(embeddings1),cluster2_count=len(embeddings2))
    return response;