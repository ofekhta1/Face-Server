from .base_request import BaseRequest
from typing import List,Annotated,Optional
from fastapi import  File, Form, UploadFile


class UploadImageRequest(BaseRequest):
    files: Annotated[List[UploadFile], File()]
    save_invalid: Annotated[Optional[bool], Form()]=False