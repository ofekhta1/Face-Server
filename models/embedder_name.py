from enum import Enum
class EmbedderName(str,Enum):
    resnet100="ResNet100GLint360K"
    resnet50="ResNet50WebFace600K"
    bb="KinshipBB"
    ss="KinshipSS"
    sibs="KinshipSIBS"
    fs="KinshipFS"
    fd="KinshipFD"
    ms="KinshipMS"
    md="KinshipMD"

    def is_kinship(embedder_name):
        return embedder_name==EmbedderName.bb or embedder_name==EmbedderName.fs