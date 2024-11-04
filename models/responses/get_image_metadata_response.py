
from typing import Annotated,Optional,List,Tuple
from fastapi import Form
from .base_response import BaseResponse
from models.face_info import FaceInfo

class GetFacesInfoResponse(BaseResponse):
    faces:List[FaceInfo]
    faces_length:int

class GetDetectorIndicesResponse(BaseResponse):
    detector_indices:dict[str,List[int]]