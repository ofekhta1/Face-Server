from .base_request import BaseRequest
from typing import Annotated,Optional
from fastapi import  Body

class AssignClusterRequest(BaseRequest):
    image:Annotated[str,Body()]
    cluster_id:Annotated[str,Body()]
    selected_face:Annotated[Optional[int],Body()]=-2