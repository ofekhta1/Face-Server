from pydantic import BaseModel
class FaceInfo(BaseModel):
    landmarks:list[list[float]]
    bbox:list[int]
    quality:float