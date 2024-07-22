from .base_request import BaseRequest
from typing import List,Annotated,Optional
from fastapi import  File, Form, UploadFile


class FilterRequest(BaseRequest):
    similarity_threshold: Annotated[float, Form(alias="similarity_thresh")]