from .base_request import BaseRequest
from typing import Annotated,Optional
from fastapi import  Body

class GetImageMetadataRequest(BaseRequest):
    image:Annotated[str,Body()]
    selected_face:Annotated[Optional[int],Body()]=-2