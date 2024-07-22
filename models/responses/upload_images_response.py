from typing import Annotated,Optional,List,Dict
from fastapi import Form
from .base_response import BaseResponse
from ..detector_name import DetectorName
class UploadImagesResponse(BaseResponse ):
    images:List[str]
    invalid_images:List[str]
    detector_indices:List[Dict[DetectorName,List[int]]]
    faces_length:List[int]