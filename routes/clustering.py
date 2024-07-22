
from fastapi import APIRouter,UploadFile,File,Body,HTTPException
from modules import AppPaths
from . import resources
import json
from models.requests import GetClustersRequest 

clustering_router=APIRouter()


@clustering_router.post("/api/cluster")
def get_groups(request:GetClustersRequest)-> dict[str, list[str]]:
    helper=resources.helper
    groups=resources.groups
    
    eps = request.max_distance
    min_samples = request.min_samples
    retrain = request.retrain
    detector_name = request.detector_name
    embedder_name = request.embedder_name

    value_groups = helper.cluster_images(
        eps, min_samples, detector_name=detector_name, embedder_name=embedder_name
    )
    if retrain:
        groups.train_index(value_groups, detector_name, embedder_name)
        groups.save_index(detector_name)
        return value_groups
    
    modified_group: dict[str, list] = {}
    if detector_name not in groups.groups or embedder_name not in groups.groups[detector_name].groups:
        return {}
     
    index = groups.groups[detector_name].groups[embedder_name].index
    for cluster_id, images in value_groups.items():
        for image in images:
            group_name = cluster_id
            if image in index:
                group_name = index[image]
            if group_name in modified_group:
                modified_group[group_name].append(image)
            else:
                modified_group[group_name] = [image]
    return modified_group


@clustering_router.post("/api/change_group_name")
def change_group_name():
    groups=resources.groups

    data = json.loads(request.get_data())
    old = data["old"]
    new = data["new"]
    detector_name = data["detector_name"] if "detector_name" in data else "SCRFD10G"
    embedder_name = (
        data["embedder_name"] if "embedder_name" in data else "ResNet100GLint360K"
    )
    if old and new and old.strip() and new.strip():
        groups.change_group_name(old, new, detector_name, embedder_name)
        groups.save_index(detector_name)
    return jsonify(success=True)
