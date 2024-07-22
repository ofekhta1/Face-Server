
from typing import Annotated,Optional,List,Dict
from fastapi import Form
from .base_response import BaseResponse
from models.similar_image import SimilarImage

class FindFaceResponse(BaseResponse):
    boxes:List[List[int]]
    faces_length:int

class GetDetectorIndicesResponse(BaseResponse):
    detector_indices:dict[str,List[int]]