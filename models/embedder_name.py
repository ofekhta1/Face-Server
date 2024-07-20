from enum import Enum
class EmbedderName(str,Enum):
    resnet100="ResNet100GLint360K"
    resnet50="ResNet50WebFace600K"
    bb="KinshipBB"
    fs="KinshipFS"