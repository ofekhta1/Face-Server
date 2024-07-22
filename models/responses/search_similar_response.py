from typing import Annotated,Optional,List,Dict
from fastapi import Form
from .base_response import BaseResponse
from models.similar_image import SimilarImage

class SearchSimilarResponse(BaseResponse):
    images:List[SimilarImage]=[]

class SearchMostSimilarResponse(BaseResponse):
    image:str
    face:int
    face_length:int
    detector_indices:dict[str, list[int]]
    similarity:float
    match:bool=True

class NoMatchResponse(BaseResponse):
    match:bool=False