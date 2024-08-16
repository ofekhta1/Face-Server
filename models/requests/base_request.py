from pydantic import BaseModel
from fastapi import   Form
from typing import Annotated
from ..detector_name import DetectorName 
from ..embedder_name import EmbedderName 

class BaseRequest(BaseModel):
    detector_name:DetectorName=DetectorName.retinaface_buffalo
    embedder_name:EmbedderName=EmbedderName.resnet100