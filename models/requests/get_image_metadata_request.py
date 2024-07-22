from .base_request import BaseRequest
from typing import Annotated
from fastapi import  Form

class GetImageMetadataRequest(BaseRequest):
    image:Annotated[str,Form()]
    selected_face:Annotated[int,Form()]=0