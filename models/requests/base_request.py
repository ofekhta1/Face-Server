from pydantic import BaseModel
from fastapi import   Form
from typing import Annotated
from ..detector_name import DetectorName 
from ..embedder_name import EmbedderName 

class BaseRequest(BaseModel):
    detector_name:Annotated[DetectorName,Form()]=DetectorName.retinaface_antelope
    embedder_name:Annotated[EmbedderName,Form()]=EmbedderName.resnet100