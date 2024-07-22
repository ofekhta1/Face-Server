from .base_request import BaseRequest
from typing import Optional,Annotated
from fastapi import Body
class GetClustersRequest(BaseRequest):
    max_distance:Annotated[Optional[float],Body()]=0.5
    min_samples:Annotated[Optional[int],Body()]=4
    retrain:Annotated[Optional[bool],Body()]=False