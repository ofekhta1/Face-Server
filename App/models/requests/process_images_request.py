from .base_request import BaseRequest
from typing import Annotated,List
from fastapi import  Body

class ProcessImagesRequest(BaseRequest):
    images:Annotated[List[str],Body()]