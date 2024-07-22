from .base_request import BaseRequest
from typing import List,Annotated
from fastapi import   Form


class CompareFacesRequest(BaseRequest):
    images: Annotated[list[str], Form()]
    selected_faces: Annotated[list[int], Form()]=False