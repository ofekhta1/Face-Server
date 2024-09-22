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


    @staticmethod
    def is_kinship(embedder_name):
        kinship_models = {EmbedderName.bb, EmbedderName.fs, EmbedderName.ss, EmbedderName.sibs, EmbedderName.fd, EmbedderName.ms, EmbedderName.md}
        return embedder_name in kinship_models
