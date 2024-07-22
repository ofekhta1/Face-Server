from typing import Annotated,Optional,List,Dict
from fastapi import Form
from .base_response import BaseResponse
from ..detector_name import DetectorName
class CompareKinshipResponse(BaseResponse):
    average_similarity:float
    cluster1_count:int
    cluster2_count:int