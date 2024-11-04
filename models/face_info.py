from typing import Optional
from pydantic import BaseModel
class FaceInfo(BaseModel):
    landmarks:list[list[float]]
    bbox:list[int]
    quality:float
    age:int
    gender:str
    group_id: Optional[str]="-1"