from typing import Annotated,Optional,List,Dict
from fastapi import Form
from .base_response import BaseResponse
from Shared.models.detector_name import DetectorName
from Shared.models.similar_image import SimilarImage
from Shared.models.face_info import FaceInfo

class SearchSimilarResponse(BaseResponse):
    images:List[SimilarImage]=[]

class SearchMostSimilarResponse(BaseResponse):
    image:str
    face:int
    face_length:int
    detector_indices:dict[str, list[int]]
    metadata:Optional[dict[str, list[FaceInfo]]]
    similarity:float
    match:bool=True

class NoMatchResponse(BaseResponse):
    match:bool=False