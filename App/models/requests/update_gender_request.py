from .base_request import BaseRequest
from typing import Annotated,Optional,Any
from fastapi import  Body

class UpdateMetadataRequest(BaseRequest):
    image:Annotated[str,Body()]
    selected_face:Annotated[int,Body()]
    metadata: Annotated[dict[str,Any],Body()]