from .base_request import BaseRequest
from typing import Optional,Annotated
from fastapi import Body
class GetClustersRequest(BaseRequest):
    max_distance:Annotated[Optional[float],Body()]=0.5
    quality_threshold:Annotated[Optional[float],Body(alias="quality_thresh",validation_alias="quality_thresh")]=0
    min_samples:Annotated[Optional[int],Body()]=4