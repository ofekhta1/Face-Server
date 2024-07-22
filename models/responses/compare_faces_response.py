from typing import Annotated,Optional,List,Dict
from fastapi import Form
from .base_response import BaseResponse
from ..detector_name import DetectorName
class CompareFacesResponse(BaseResponse):
    similarity:float
