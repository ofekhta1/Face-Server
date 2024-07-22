from typing import Optional,Annotated
from pydantic import BaseModel
class BaseError(BaseModel):
    reason:str=""