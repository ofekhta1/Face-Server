
from flask import Blueprint, request, jsonify
from modules import AppPaths,FamilyClassifier
from . import resources
import json


image_clustering_bp = Blueprint('Clustering', __name__)


@image_clustering_bp.route("/api/cluster", methods=["POST"])
def get_groups():
    helper=resources.helper
    groups=resources.groups
    
    jsonData = request.get_data()
    data = json.loads(jsonData) if jsonData else {}
    eps = float(data["max_distance"]) if "max_distance" in data else 0.5
    min_samples = int(data["min_samples"]) if "min_samples" in data else 4
    retrain = data["retrain"] if "retrain" in data else False
    detector_name = data["detector_name"] if "detector_name" in data else "SCRFD10G"
    embedder_name = (
        data["embedder_name"] if "embedder_name" in data else "ResNet100GLint360K"
    )
    classifier = FamilyClassifier(AppPaths.APP_DIR)

    cluster_family = data.get("cluster_family", False)
    if cluster_family:
        value_groups = helper.cluster_images_family(
            eps,
            min_samples,
            detector_name=detector_name,
            embedder_name=embedder_name,
            classifier=classifier,
        )
    else:
        value_groups = helper.cluster_images(
            eps, min_samples, detector_name=detector_name, embedder_name=embedder_name
        )
    if retrain:
        groups.train_index(value_groups, detector_name, embedder_name)
        groups.save_index(detector_name)
        return jsonify(value_groups)
    modified_group: dict[str, list] = {}
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
    return jsonify(modified_group)


@image_clustering_bp.route("/api/change_group_name", methods=["POST"])
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
