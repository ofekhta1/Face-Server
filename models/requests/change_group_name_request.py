from .base_request import BaseRequest
from typing import List,Annotated
from fastapi import Body


class ChangeGroupNameRequest(BaseRequest):
    old: Annotated[str, Body()]
    new: Annotated[str, Body()]=False